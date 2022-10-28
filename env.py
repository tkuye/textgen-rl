"""
Module for creating openai gym compatible environment for text generation
"""

import gym
import numpy as np
import torch
VOCAB_SIZE = 50257

class TextGym(gym.Env):
    def __init__(self, size=VOCAB_SIZE, max_length=512, tokenizer=None, reward_fn=None):
        self.size = size
        self.action_space = gym.spaces.Text(max_length=max_length)
        self.observation_space = gym.spaces.Text(max_length=max_length)
        self.state = 0
        self.reward = 0
        self.done = False
        self.info = {}
        self.max_length = max_length
        self.current_length = 0
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        
    def step(self, action):
        self.state = torch.cat((self.state, action.unsqueeze(-1).unsqueeze(-1)), dim=-1)
        self.current_length += self.state.size()[-1]
        if self.current_length >= self.max_length:
            self.done = True
        self.reward = self.reward_fn(action, self.info)
        
        return self.state, self.reward, self.done, self.info
    
    def reset(self):
        self.state = self.tokenizer("hello", return_tensors='pt').input_ids
        self.reward = 0
        self.done = False
        self.info = {}
        self.current_length = 0
        return self.state

    def render(self, mode='human'):
        return self.state

    
    def close(self):
        pass