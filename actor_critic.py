import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from buffer import ReplayBuffer 
from model import Policy, QNetwork


class SoftActorCritic:
    def __init__(self, epochs, gamma, env, batch_size, buffer_size, policy_lr, qf_lr, policy_file_path, qf_file_path):
        self.epochs = epochs
        self.gamma = gamma

        ## Our Q functions
        self.actor = QNetwork()
        self.critic = QNetwork()
        self.env = env
        self.buffer = ReplayBuffer(
            buffer_size, self.env.observation_space.shape, self.env.action_space.shape
        )
        ## Our policy
        self.policy = Policy()
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.qf_optim = torch.optim.Adam(self.policy.parameters(), lr=qf_lr)

        self.policy_file_path = policy_file_path
        self.qf_file_path = qf_file_path

        self.policy.load_state_dict(torch.load(self.policy_file_path))
        self.critic.load_state_dict(torch.load(self.qf_file_path))
       
        self.batch_size = batch_size
        

    def act(self, state):
        action, _ = self.policy.sample(state)
        return action

    def save(self, path):
        torch.save(self.policy.model.state_dict(), path)

    def update(self):
        for _ in range(self.epochs):
            states, actions, rewards, states_, dones = self.buffer.sample_buffer(self.batch_size)
            ## update Q functions
            
            with torch.no_grad():
                next_actions, next_logprobs = self.policy.sample(states_)
                q1_next, q2_next = self.critic.forward(next_actions, states_)
                q_next = torch.min(q1_next, q2_next) - next_logprobs
                q_target = rewards + (1 - dones) * self.gamma * q_next
            
            q1, q2 = self.actor.forward(actions, states)
            qf_loss = nn.MSELoss(q1, q_target) + nn.MSELoss(q2, q_target)
            self.qf_optim.zero_grad()
            qf_loss.backward()
            ## update policy
            actions, logprobs = self.policy.sample(states)
            q1_pi, q2_pi = self.critic.forward(actions, states)
            q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = (logprobs - q_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
            self.qf_optim.step()

    def learn(self):
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            action = self.act(state)
            state_, reward, done, _ = self.env.step(action)
            self.buffer.store_transition(state, action, reward, state_, done)
            state = state_
            total_reward += reward
            if self.buffer.mem_cntr > self.batch_size:
                self.update()
        return total_reward

    def train(self):
        total_rewards = []
        for i in range(self.epochs):
            total_reward = self.learn()
            total_rewards.append(total_reward)
            if i % 10 == 0:
                print("Epoch: {}, Total Reward: {}".format(i, total_reward))
                self.save_models()
        return total_rewards


    def save_models(self):
        print("... saving models ...")
        self.policy.save(self.policy_file_path)
        self.critic.save(self.qf_file_path)
    





