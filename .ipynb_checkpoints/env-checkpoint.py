"""
Module for creating openai gym compatible environment for text generation
"""

import gym
import numpy as np
import torch
import re
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
        
        state_text = self.tokenizer.batch_decode(self.state)[0]
        action_text = self.tokenizer.batch_decode(action)[0]
        state_text = re.sub('<pad>', '', state_text)
        state_text += action_text
        self.current_length = self.tokenizer(state_text, return_tensors='pt').input_ids.size()[-1]
        self.state = self.tokenizer(state_text, padding='max_length', max_length=512, return_tensors='pt').input_ids
        if self.current_length >= self.max_length:
            self.done = True
        self.reward = self.reward_fn(self.state, self.info)
        
        return self.state, self.reward, self.done, self.info
    
    def reset(self):
        self.state = self.tokenizer("hello", padding='max_length', max_length=512, return_tensors='pt').input_ids
        self.reward = 0
        self.done = False
        self.info = {}
        self.current_length = 0
        return self.state

    def render(self, mode='human'):
        return self.state

    
    def close(self):
        pass