## Replay Buffer Code
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        dtype = np.int8 if n_actions < 256 else np.int16
        self.action_memory = np.zeros(self.mem_size, dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def __len__(self):
        return self.mem_cntr

    def save(self, path):
        np.savez(path, state_memory=self.state_memory,
                 new_state_memory=self.new_state_memory,
                 action_memory=self.action_memory,
                 reward_memory=self.reward_memory,
                 terminal_memory=self.terminal_memory)

    def load(self, path):
        data = np.load(path)
        self.state_memory = data['state_memory']
        self.new_state_memory = data['new_state_memory']
        self.action_memory = data['action_memory']
        self.reward_memory = data['reward_memory']
        self.terminal_memory = data['terminal_memory']
        self.mem_cntr = len(self.state_memory)

    def clear(self):
        self.state_memory = np.zeros((self.mem_size, *self.state_memory.shape[1:]),
                                     dtype=np.float32)