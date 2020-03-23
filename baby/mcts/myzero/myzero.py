"""
MyZero is a simple MCTS implementation compatible with OpenAI gym env

selection - expansion - simulation - backpropagation

Inspired by MuZero:
https://medium.com/applied-data-science/how-to-build-your-own-deepmind-muzero-in-python-part-2-3-f99dad7a7ad

"""

from baby.mcts.myzero.node import Node
from baby.mcts.myzero.minmaxstats import MinMaxStats

default_conf = {
    'num_simulations': 1
}

class MyZero:
    def __init__(self):
        self.conf = default_conf
        self.min_max_stats = MinMaxStats()


    def run(self):
        for _ in range(self.conf['num_simulations']):
            history = action_history.clone()
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.selection(
                    config,
                    node,
                    self.min_max_stats
                )
                history.add_action(action)

            self.expand_node(node, history.to_play(), history.action_space(), network_output)

            self.backpropagate(search_path, network_output.value, history.to_play(),
                        config.discount, self.min_max_stats)

    def selection(self):
        pass

    def expand_node(self):
        pass

    def simulation(self):
        pass

    def backpropagate(self):
        pass