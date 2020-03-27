import numpy as np

class ReplayBuffer:
    def __init__(self, size: int=1000):
        """
        ReplayBuffer of data
        """
        self.buffer = []
        self.size = size

    def sample(self):
        return np.random.choice(self.buffer)

    def add(self, element):
        self.buffer.append(element)

        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def batch(self, n_batch: int):
        batch = [self.sample() for _ in range(n_batch)]
        return batch

