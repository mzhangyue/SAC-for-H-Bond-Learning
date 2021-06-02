import random
import numpy as np

'''
This class implements a replay buffer with a Python list as the internal data
structure. The entries store in the replay buffer must be of the form 
(state, action, reward, next_state, done) where state, action, reward, and 
next_state are vectors and done is a bool. When capacity of the replay buffer
is reached, future entries override the earlier entries.
'''

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # Pushes one state, action, reward, next_state, done to the replay buffer
    # At capacity, we override earlier entries
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # Sample a batch of (state, action ,reward, next_state, done)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)