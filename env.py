"""
Module for creating openai gym compatible environment for text generation
"""

import gym
import numpy as np
import torch
import re
import random

VOCAB_SIZE = 32128

class TextGym(gym.Env):
    def __init__(self, size=VOCAB_SIZE, max_length=512, tokenizer=None, reward_fn=None):
        self.size = size
        self.action_space = gym.spaces.Box(low=0, high=self.size - 1, shape=(512,), dtype=np.int64)
        self.observation_space = gym.spaces.Box(low=0, high=self.size, shape=(512,), dtype=np.int64)
        self.state = 0
        self.reward = 0
        self.done = False
        self.info = {}
        self.max_length = max_length
        self.current_length = 0
        self.steps = 0 
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        
    def step(self, action, rand_action_prob=0.1):
        ## We take a random action with probability rand_action_prob
        self.state = action
        self.current_length = torch.nonzero(self.state).size(-1)
        if self.current_length >= self.max_length or self.steps > 512:
            self.done = True
        observation = self._get_obs(rand_action_prob=rand_action_prob)
        self.reward = self.reward_fn(observation, self.info)
        self.steps += 1
        return observation, self.reward, self.done, self.info
    
    def reset(self):
        self.state = self.tokenizer("hello world", padding='max_length', max_length=512, return_tensors='pt').input_ids
        self.reward = 0
        self.done = False
        self.info = {}
        self.current_length = 0
        return self.state[0]

    def render(self, mode='human'):
        return self.state


    def _get_obs(self, rand_action_prob=0.1):
        
        
        if random.random() < rand_action_prob:
            action = self.action_space.sample()
            
            return torch.tensor(action)
        return self.state

    
    def close(self):
        pass