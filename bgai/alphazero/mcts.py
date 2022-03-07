
import math
import scipy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

from bgai.santorini import BOARD_SHAPE, BOARD_SIZE, Santorini
from bgai.visualize import render_path

import logging
log = logging.getLogger(__name__)

SIMULATIONS = 100
NUM_SAMPLING_MOVES = 0

C_BASE = 19652
C_INIT = 0

ROOT_DIRICHLET_ALPHA = 0.3
ROOT_EXPLORATION_FRACTION = 0.25


@dataclass
class Node:
    game:           Santorini
    terminal:       bool        = False
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


def mcts(game: Santorini):
    root = Node(game)
    expand(root, add_exploration_noise=True)

    _path_depth_sum = 0
    for s in range(SIMULATIONS):
        path = [root]
        node = root

        while not node.is_leaf:
            # Do we even need to have a map for the childrens variable?
            node = max(node.children.values(), key=lambda child: ucb(node, child))
            path.append(node)

        value = expand(node)
        expanded_player_id = node.game.current_player_id
        _path_depth_sum += len(path)

        for n in path:
            n.visit_count += 1
            n.value_sum += value if n.game.current_player_id == expanded_player_id else -value
        
        if log.isEnabledFor(logging.DEBUG) and (s + 1) % (SIMULATIONS // 10) == 0 :
            log.debug(f"MCTS SIM {s} | Best child value {max(map(lambda c: c.value, root.children.values())):.3f} | Average depth {_path_depth_sum / (s + 1): .3f} | Most visited {max(map(lambda c: c.visit_count, root.children.values())) / (s + 1):.3f} ({len(root.children)})")

    if game.turn < NUM_SAMPLING_MOVES:
        # Select action proportional to softmax of visit count
        actions, visit_counts = zip(*tuple((action, child.visit_count) for action, child in root.children.items()))
        action = actions[np.random.choice(len(actions), p=scipy.special.softmax(visit_counts))]
    else:
        # Select the action that was visited most often
        action = max(root.children.keys(), key=lambda action: root.children[action].visit_count)
    
    return action, root


def expand(node: Node, add_exploration_noise: bool = False):
    #TODO: Actually get prediction and legal actions. (Represent legal actions as 2* 4d?)
    value = 1.0 if node.terminal else 0.0
    policy_logits = np.ones(shape=(2, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.int_)

    if not node.terminal:
        # Softmax applied only over legal moves
        policy = {action: math.exp(policy_logits[action.as_tuple(node.game)]) for action in node.game.get_legal_actions()}
        policy_sum = sum(policy.values())

        if policy:
            for action, p in policy.items():
                prior = p / policy_sum

                if add_exploration_noise:
                    prior *= 1 - ROOT_EXPLORATION_FRACTION
                    prior += np.random.gamma(ROOT_DIRICHLET_ALPHA, 1, 1) * ROOT_EXPLORATION_FRACTION

                node.children[action] = Node(
                    game=node.game.apply_legal_action(action),
                    terminal=node.game.is_winning_action(action),
                    prior=prior
                )
        else:
            raise ValueError("Did not find any legal actions in non-terminal node")
    return value


def ucb(parent: Node, child: Node):
    exploration_rate = math.log(1 + (parent.visit_count + 1) / C_BASE) + C_INIT
    u = exploration_rate * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)

    return child.value + u
