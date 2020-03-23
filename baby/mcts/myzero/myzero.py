"""
MyZero is a simple MCTS implementation compatible with OpenAI gym env

selection - expansion - simulation - backpropagation

Inspired by MuZero:
https://medium.com/applied-data-science/how-to-build-your-own-deepmind-muzero-in-python-part-2-3-f99dad7a7ad

"""
from typing import List

import numpy as np

from baby.mcts.myzero.node import Node

from baby.heuristic.greedy import GreedyHeuristic

default_conf = {
    # Number of simulations
    'num_simulations': 100,
    # Number of action chosen during 'expand_node'
    'n_exploration': 10,
    # Exploration factor (c)
    'exploration_factor': 1.41,
    # Gamma factor
    'gamma': 0.995,
}

class MyZero:
    def __init__(self, env):
        self.conf = default_conf
        self.env = env
        self.policy_exploration = GreedyHeuristic(n=self.conf['n_exploration'])
        self.policy_simulation = GreedyHeuristic(n=1)

        # Specific reward system for MCTS
        self.env.conf['reward'] =  {
            # Reward at each step
            'timestep': 0,
            # Reward for each validated element
            'validation': 0,
            # Reward at completion (episode done)
            'done': 1,
        }


    def run(self):
        """
        Train MCTS
        """
        node = self.root()
        # Store search path for backpropagation
        search_path = [node]

        for _ in range(self.conf['num_simulations']):
            # Select node children using UCB scoring unti a leaf is reached
            while node.expanded():
                action, node = self.selection(node)
                search_path.append(node)

            # Execute Monte-Carlo simulation to estimate children values
            self.expand_node(node, self.conf['n_exploration'])

            # Simulation
            self.simulation(node)

            # Backpropagate Q-values
            self.backpropagate(search_path)

            print(f"#{search_path[0].visit_count} // Score root = {search_path[0].reward}")
    

    def root(self):
        """
        Generate root node
        """
        # Reset environment
        self.env.reset()
        # Create root node
        root = Node(self.env)

        return root

    def selection(self, node):
        """
        Start from root node, reach leaf by using UCB score
        """
        node_children = node.children
        ucb_scores = np.array([self.ucb(node_child, node) for node_child in node_children])

        action = np.argmax(ucb_scores)
        child = node_children[action]

        return action, child

    def expand_node(self, node, n_exploration):
        """
        Define n children at node
        """
        # Prepare output
        children = []

        # Use heuristic to determine n action
        c_obs = node.get_state().current_obs

        # For each action, create a child node
        for action in self.policy_exploration.act(c_obs):
            # Load current enœv state and act
            env_child = node.get_state()
            env_child.step(action)

            # print(f"Take {action} action for env {env_child}")
            
            child = Node(env_child)
            children.append(child)

        # Store children
        node.children = children

    def simulation(self, node: Node):
        pi_simu = self.policy_simulation

        for idx, child in enumerate(node.children):
            env = child.get_state()
            obs = env.current_obs
            done = False

            while not done:
                action = pi_simu.act(obs)
                obs, rew, done, info = env.step(action)
            
            # Reward at leaf is gamma power timestep
            child.reward = self.conf['gamma'] ** env.t
            child.cum_rewards = child.reward
            child.visit_count += 1

            # print(f"Child #{idx} ends after {env.t} steps")

    def backpropagate(self, search_path: List):
        # From leaf node to root
        for node in search_path[::-1]:
            best_reward_children = np.max([child.reward for child in node.children])
            node.cum_rewards += best_reward_children
            node.reward = best_reward_children # Optionnal ??
            node.visit_count += 1


        # for node in search_path:
        #     print(f"Node {node.value()} / {node.visit_count}")

    def ucb(self, node: Node, node_parent: Node):
        return node.value() + self.conf['exploration_factor'] * np.sqrt(node_parent.visit_count / node.visit_count)
