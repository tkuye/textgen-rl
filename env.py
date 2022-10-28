"""
Module for creating openai gym compatible environment for text generation
"""

import gym
VOCAB_SIZE = 50257
class TextGym(gym.Env):
    def __init__(self, size=VOCAB_SIZE, max_length=512, tokenizer=None):
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
    
    def step(self, action, reward_fn):
        self.current_length += len(action)
        if self.current_length >= self.max_length:
            self.done = True
        self.state = action
        self.reward = reward_fn(self.state, self.info)
        return self.state, self.reward, self.done, self.info
    
    def reset(self):
        self.state = 0
        self.reward = 0
        self.done = False
        self.info = {}
        self.current_length = 0
        return self.state

    def render(self, mode='human'):
        return self.state

    
    def close(self):
        pass