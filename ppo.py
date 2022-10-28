import torch
from transformers import T5ForConditionalGeneration, BertForSequenceClassification
from torch.distributions import Categorical
from rl_model import BaseModel
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
class PPO(BaseModel):
    """
    PPO Class implementation for use with huggingface transformers.
    """
    def __init__(self, epochs=100, gamma=0.1, env=None, batch_size=32, clip_param=1.0, policy_epochs=10000, value_epochs=10000, lr=1e-4, betas=1.0, eps=1.0, max_grad_norm=1.0, max_len=512, filename="ppo", train=True):
        self.epochs = epochs
        self.gamma = gamma
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        self.critic = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.policy_epochs = policy_epochs
        self.value_epochs = value_epochs
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.filename = filename
        self.max_len = 512
        if train:
            self.train()

    def to_text(self, input_ids):
        return self.env.tokenizer.decode(input_ids, skip_special_tokens=True)

    def act(self, state):
        next_state = self.shift_right(state)
        next_state = next_state.to(self.device)
        state = state.to(self.device)
        
        outputs = self.actor(input_ids=state, decoder_input_ids=next_state)
        logits = outputs.logits
        logits_softmax = F.softmax(logits, dim=-1)
        action = logits_softmax.argmax()
        action_logprob = logits_softmax.max()
        return action.detach().cpu(), action_logprob.detach().cpu()

    
    def rollout(self):
        state = self.env.reset()
        memory = []
        for _ in range(self.max_len):
            
            action, action_logprob = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            memory.append([state, action, action_logprob, reward, next_state, done])
            state = next_state
            if done:
                break
        return memory
    
    def shift_right(self, input_ids):
        decoder_start_token_id = self.env.tokenizer.pad_token_id
        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"
        
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        return shifted_input_ids
    
   
    def train(self):
        print("Starting training..")
        for _ in tqdm(range(self.epochs)):
            memory = self.rollout()
            states, actions, action_logprobs, rewards, next_states, dones = zip(*memory)
            
            actions = actions.to(self.device)
            action_logprobs = action_logprobs.to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            next_states = next_states.to(self.device)
            dones = torch.tensor(dones).to(self.device)
            returns = self.compute_returns(rewards, dones)
            advantages = returns - self.critic(input_ids=states, decoder_input_ids=self.shift_right(states)).logits

            for _ in range(self.policy_epochs):
                _, new_action_logprobs = self.act(states)
                ratio = (new_action_logprobs - action_logprobs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                self.actor_optim.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

            for _ in range(self.value_epochs):
                values = self.critic(input_ids=states, decoder_input_ids=self.shift_right(states)).logits
                value_loss = (returns - values).pow(2).mean()
                self.critic_optim.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

            self.save()
    
    def compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns

    
    
    def save(self):
        print("Saving model..")
        torch.save(self.actor.state_dict(), self.filename + "_actor")
        torch.save(self.critic.state_dict(), self.filename + "_critic")


    @classmethod
    def load(cls, filename):
        model = cls(train=False)
        model.actor.load_state_dict(torch.load(filename + "_actor"))
        model.critic.load_state_dict(torch.load(filename + "_critic"))
        return model


