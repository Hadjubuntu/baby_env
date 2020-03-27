"""
MyZero is a simple MCTS implementation compatible with OpenAI gym env

selection - expansion - simulation - backpropagation

Inspired by MuZero:
https://medium.com/applied-data-science/how-to-build-your-own-deepmind-muzero-in-python-part-2-3-f99dad7a7ad

"""
from typing import List

import numpy as np

from baby.mcts.myzero.node import Node
from baby.mcts.myzero.replay_buffer import ReplayBuffer
from baby.mcts.myzero.model_value import ModelValue

from baby.heuristic.greedy import GreedyHeuristic

default_conf = {
    # Number of simulations
    'num_simulations': 1000,
    # Number of action chosen during 'expand_node'
    'n_exploration': 3,
    # Exploration factor (c)
    'exploration_factor': 0.0001,
    # Gamma factor
    'gamma': 0.997,
    # Exploration method 
    # - FULL: add all legal actions to node,
    # - RAND: add random legal action to node
    # - DIST: add legal action with random selection according to probability distributioon
    'exploration_method': 'FULL',
    # Replay buffer size
    'replay_buffer_size': 1000,
    # Value model batch size
    'batch_size': 32,

}

class MyZero:
    def __init__(self, env):
        self.conf = default_conf

        self.env = env
        self.policy_exploration = GreedyHeuristic(n=self.conf['n_exploration'])
        self.policy_simulation = GreedyHeuristic(n=1)
        self.replay_buffer = ReplayBuffer(size=self.conf['replay_buffer_size'])
        self.model_value = ModelValue(
            input_shape=self.env.observation_space.shape,
            batch_size=self.conf['batch_size']
        )

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
        for n_simu in range(self.conf['num_simulations']):
            next_node = node

            # Store search path for backpropagation
            search_path = [next_node]

            # Select node children using UCB scoring unti a leaf is reached
            while next_node.expanded() and not next_node.terminal():
                action, next_node = self.selection(next_node)
                search_path.append(next_node)

            if not next_node.terminal():
                # Execute Monte-Carlo simulation to estimate children values
                self.expand_node(next_node, self.conf['n_exploration'])

                # Simulation
                self.simulation(next_node)

                # Backpropagate Q-values
                self.backpropagate(search_path)

                # TODO: Improve architecture MCTS, follows train/collect/exploit philosophy of AlphaZero
                batch = self.replay_buffer.batch(n_batch=self.conf['batch_size'])
                mse_value = self.model_value.train(batch)

                t = np.min([child.infos['timesteps'] for child in search_path[-1].children.values()])
                print(f"#{search_path[0].visit_count} // Score root = {search_path[0].value()} // timesteps={t} // search_path_length={len(search_path)} // mse={mse_value}")
            else:
                # Simulation
                self.simulation(next_node)
                print("Revisit terminal node")
             

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
        ucb_scores = np.array([self.ucb(node_child, node) for node_child in node_children.values()])

        action = np.argmax(ucb_scores)
        child = list(node_children.values())[action]

        return action, child

    def expand_node(self, node, n_exploration):
        """
        Define n children at node
        """
        # Prepare output
        children = {}
        
        #  Restore current env state
        c_env = node.get_state(self.env)

        # Use heuristic to determine n action
        c_obs = c_env.current_obs

        action_legals = self.policy_exploration.act(c_obs)
        action_chosen = self.select_action(action_legals)

        # For each action chosen, create a child node
        for action in action_chosen:
            # Load current env state and act
            c_env.step(action)
            
            child = Node(c_env)
            children[action] =  child

        # Store children
        node.children = children

    def simulation(self, node: Node):
        pi_simu = self.policy_simulation
        best_node_score = None

        for child in node.children.values():
            # Retrieve env state            
            env = child.get_state(self.env)
            obs = env.current_obs
            done = child.done

            # And enjoy environment until its done with simulation policy            
            while not done:
                action = pi_simu.act(obs)
                obs, rew, done, info = env.step(action)               
                
            # Reward at leaf is gamma power timestep
            child.reward = self.conf['gamma'] ** env.t
            child.cum_rewards = child.reward
            child.visit_count = 1
            child.infos['timesteps'] = env.t

            if not best_node_score or child.reward >= best_node_score:
                best_node_score = child.reward

        # Optimist view: take best children reward of leaf 
        node.reward = best_node_score


    def backpropagate(self, search_path: List):
        # From leaf node to root
        last_value = search_path[-1].reward

        for node in search_path[::-1]:
            node.cum_rewards += last_value
            node.visit_count += 1

            # Add those data to replay buffer
            # And enjoy model to see accuracy
            origin_obs = np.copy(node.get_state(self.env).current_obs)
            model_value = self.model_value.predict(np.array([origin_obs])).numpy()
            print(f"model_value={np.squeeze(model_value)} vs true_value={node.value()}")
            self.replay_buffer.add({'obs': origin_obs, 'value': node.value()})


    def ucb(self, node: Node, node_parent: Node):
        explo_factor = self.conf['exploration_factor'] * np.sqrt(node_parent.visit_count / node.visit_count)
        return node.value() + explo_factor


    def select_action(self, action_legals):
        """
        Select actions that will led to children creation
        """
        if self.conf['exploration_method'] == 'FULL':
            return action_legals
        elif self.conf['exploration_method'] == 'RAND':
            action_idx = np.random.randint(len(action_legals))
            return np.array([action_legals[action_idx]])
        else:
            raise NotImplemented('Not yet implemented')
        
