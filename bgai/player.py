from multiprocessing.sharedctypes import Value
import random
from bgai.alphazero.mcts import mcts
from bgai.santorini import Santorini

import logging
log = logging.getLogger(__name__)


class BasePlayer:
    player_type = "base"

    def __init__(self, id: int, marker: str):
        self._id = id
        self._marker = marker
    
    def __str__(self):
        return f"Player '{self.player_type}' marked as '{self._marker}'"


class RandomPlayer(BasePlayer):
    player_type = "random"

    def get_action(self, game: Santorini):
        return random.choice(tuple(game.get_legal_actions()))


class RandomFinisherPlayer(BasePlayer):
    player_type = "random-finisher"

    def get_action(self, game: Santorini):
        for action in game.get_legal_actions():
            if game.is_winning_action(action):
                return action
        
        return random.choice(tuple(game.get_legal_actions()))


class ClimberPlayer(BasePlayer):
    player_type = "climber"

    def get_action(self, game: Santorini):
        actions = filter(lambda a: game._board[a.worker] < game._board[a.destination], game.get_legal_actions())

        try:
            return max(actions, key=lambda a: game._board[a.build] - 3 * (game._board[a.build] > game._board[a.destination]))
        except ValueError:
            return random.choice(tuple(game.get_legal_actions()))


class MctsPlayer(BasePlayer):
    player_type = 'mcts'

    def get_action(self, game: Santorini):
        return mcts(game)


class InputPlayer(BasePlayer):
    player_type = "input"

    def get_action(game: Santorini):

        # Retrieve valid action from input()

        return 


PLAYER_TYPES = { c.player_type: c for c in (InputPlayer, RandomPlayer, RandomFinisherPlayer, ClimberPlayer, MctsPlayer)}
