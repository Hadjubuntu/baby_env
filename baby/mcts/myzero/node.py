"""
Node class
"""
class Node():    
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return (len(self.children) > 0)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count