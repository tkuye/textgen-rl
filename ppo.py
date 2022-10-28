import torch
from transformers import T5ForConditionalGeneration, BertForSequenceClassification
from torch.distributions import Categorical
from rl_model import BaseModel
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn



class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        self.critic = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1).to(self.device)
        

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = state.unsqueeze(0)
        next_state = self.actor._shift_right(state)
        next_state = next_state.to(self.device)
        state = state.to(self.device)
        attention_masks = torch.where(next_state != 0, 1, 0)
        outputs = self.actor(input_ids=state, decoder_input_ids=next_state, attention_mask=attention_masks)
        logits = outputs.logits
        logits = logits[-1, :, :] 
        logits = self.top_k_logits(logits, 1)
        log_probs = F.softmax(logits, dim=-1)
        action_logprob, action = log_probs.max(dim=-1)
        return action.cpu(), action_logprob.cpu()

    
    def evaluate(self, state, action):
        next_state = self.actor._shift_right(state)
        next_state = next_state.to(self.device)
        state = state.to(self.device)
        attention_masks = torch.where(next_state != 0, 1, 0)
        outputs = self.actor(input_ids=state, decoder_input_ids=next_state, attention_mask=attention_masks)
        logits = outputs.logits
        logits = logits[:, -1, :] 
        logits = self.top_k_logits(logits, 1)
        log_probs = F.softmax(logits, dim=-1)
        action_logprob, action = log_probs.max(dim=-1)
        state = state * (state <= self.critic.config.vocab_size)
        state_logits = self.critic(input_ids=state, attention_mask=attention_masks).logits.detach().cpu()
        return action_logprob.detach().cpu(), state_logits


    def top_k_logits(self, logits, k):
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1]
        return torch.where(logits < min_values.view(logits.size(0), -1), torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


class PPO(BaseModel):
    """
    PPO Class implementation for use with huggingface transformers.
    """
    def __init__(self, epochs=100, gamma=0.1, env=None, batch_size=32, clip_param=1.0, lr=1e-4, max_len=512, filename="ppo", train=True, timesteps=10000):
        self.epochs = epochs
        self.gamma = gamma
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy()
        self.old_policy = Policy()
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr},
                        {'params': self.policy.critic.parameters(), 'lr': lr}
                    ])
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.lr = lr
        self.filename = filename
        self.max_len = max_len
        self.loss = nn.MSELoss()
        self.timesteps = timesteps
        self.ep_reward = 0 
        if train:
            self.train()

    def rollout(self):
        state = self.env.reset()
        memory = []
        for _ in tqdm(range(self.max_len)):
            action, action_logprob = self.old_policy.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.ep_reward += reward
            memory.append([state, action, action_logprob, reward, done])
            state = next_state
            if done:
                break
        return memory


    def update(self):
        memory = self.rollout()
        old_states, old_actions, old_logprobs, rewards, dones = zip(*memory)
        returns = self.compute_returns(rewards, dones)
        old_states = torch.stack(list(old_states), dim=1).squeeze(0).to(self.device)
        old_logprobs = torch.stack(list(old_logprobs)).cpu()
        old_actions = torch.stack(list(old_actions)).cpu()
        rewards = torch.tensor(list(rewards)).cpu() 

        for _ in range(self.epochs):
            logprobs, state_values = self.policy.evaluate(old_states, old_actions)
            advantages = returns - state_values
            
            advantages = (advantages - advantages.mean(dtype=torch.float32)) / (advantages.std() + 1e-5)
            ratio = (logprobs - old_logprobs).exp().view(-1, 0)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(torch.squeeze(state_values), rewards)
            loss.requires_grad = True
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            ## Copy new weights to old policy.
        self.old_policy.load_state_dict(self.policy.state_dict())
        

    def train(self, show_steps=3):
        print_running_reward = 0
        print_running_episodes = 0

        for i in tqdm(range(1, self.timesteps + 1)):
            self.update()
            if i % show_steps == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("\n\nEpisode : {} \t\t Average Reward : {}".format(print_running_episodes, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0
                self.save()

            print_running_reward += self.ep_reward
            print_running_episodes += 1
            self.ep_reward = 0
    
    def compute_returns(self, rewards, dones):
        returns = torch.zeros_like(torch.tensor(list(rewards)))
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns

    
    def save(self):
        print("\nSaving model..\n")
        torch.save(self.policy.actor.state_dict(), self.filename + "_actor")
        torch.save(self.policy.critic.state_dict(), self.filename + "_critic")


    @classmethod
    def load(cls, filename):
        model = cls(train=False)
        model.actor.load_state_dict(torch.load(filename + "_actor"))
        model.critic.load_state_dict(torch.load(filename + "_critic"))
        return model


