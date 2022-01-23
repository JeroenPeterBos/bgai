
import math
import scipy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

from bgai.santorini import Santorini


SIMULATIONS = 1000
NUM_SAMPLING_MOVES = 0

C_BASE = 19652
C_INIT = 1.25

ROOT_DIRICHLET_ALPHA = 0.3
ROOT_EXPLORATION_FRACTION = 0.25


@dataclass
class Node:
    game:           Santorini
    prior:          float       = 0
    visit_count:    int         = 0
    value_sum:      float       = 0.0
    children:       Dict        = field(default_factory=dict)

    @property
    def is_leaf(self):
        return not len(self.children) > 0
    
    @property
    def value(self):
        if self.visit_count > 0:
            return self.value_sum / self.visit_count
        else:
            return 0


def mcts(game: Santorini, turn: int = 0):
    root = Node(game)
    expand(root, add_exploration_noise=True)

    for _ in range(SIMULATIONS):
        node = root
        path = [node]

        while not node.is_leaf:
            # Do we even need to have a map for the childrens variable?
            node = max(node.children.values(), key=lambda child: ucb(node, child))
            path.append(node)

        value = expand(node)
        for node in path:
            node.visit_count += 1
            node.value_sum += value
    
    if turn < NUM_SAMPLING_MOVES:
        # Select action proportional to softmax of visit count
        actions, visit_counts = zip(*tuple((action, child.visit_count) for action, child in root.children.items()))
        return actions[np.random.choice(len(actions), p=scipy.special.softmax(visit_counts))]
    else:
        # Select the action that was visited most often
        return max(root.children.items(), lambda candidate: candidate[1].visit_count)[0]


def expand(node: Node, add_exploration_noise: bool = False):
    #TODO: Actually get prediction and legal actions. (Represent legal actions as 2* 4d?)

    policy = {}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        prior = p / policy_sum

        if add_exploration_noise:
            prior *= 1 - ROOT_EXPLORATION_FRACTION
            prior += np.random.gamma(ROOT_DIRICHLET_ALPHA, 1, 1) * ROOT_EXPLORATION_FRACTION

        node.children[action] = Node(node.game.apply_action(action), prior)
    return 0


def ucb(parent: Node, child: Node):
    exploration_rate = math.log((1 + parent.visit_count) + C_BASE) + C_INIT
    u = child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return child.value + exploration_rate * u
