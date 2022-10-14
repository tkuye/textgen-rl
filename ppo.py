import torch
from transformers import T5ForConditionalGeneration, BertForSequenceClassification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical

class PPO:
    """
    PPO Class implementation for use with huggingface transformers.
    """
    def __init__(self, epochs, gamma, env, batch_size, clip_param, policy_epochs, value_epochs, lr, betas, eps, max_grad_norm, entropy_coef, value_loss_coef, max_steps):
        self.epochs = epochs
        self.gamma = gamma
        self.env = env
        self.actor = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        self.critic = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
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
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_steps = max_steps


    def act(self, state):
        outputs = self.actor(state)
        logits = outputs.logits
        dist = Categorical(logits=logits)
        action = dist.rsample()
        action_logprob = dist.log_prob(action)
        return action.detach().numpy(), action_logprob.detach()

    

    def rollout(self):
        state = self.env.reset()
        memory = []
        for _ in range(self.max_steps):
            action, action_logprob = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            memory.append((state, action, action_logprob, reward, next_state, done))
            state = next_state
            if done:
                break
        return memory

    
    def update(self):
        for _ in range(self.epochs):
            memory = self.rollout()
            states, actions, action_logprobs, rewards, next_states, dones = zip(*memory)
            states = torch.tensor(states).to(device)
            actions = torch.tensor(actions).to(device)
            action_logprobs = torch.tensor(action_logprobs).to(device)
            rewards = torch.tensor(rewards).to(device)
            next_states = torch.tensor(next_states).to(device)
            dones = torch.tensor(dones).to(device)
            returns = self.compute_returns(rewards, dones)
            advantages = returns - self.critic(states).logits
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
                values = self.critic(states).logits
                value_loss = (returns - values).pow(2).mean()
                self.critic_optim.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
    
    def compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns

    
    def train(self):
        for epoch in range(self.epochs):
            self.update()
            print("Epoch: {}".format(epoch))
        self.save("ppo")
    
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")


