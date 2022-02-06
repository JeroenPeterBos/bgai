import random
from tokenize import Name
from typing import NamedTuple, Tuple
from itertools import combinations, product
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


SAFE_MODE = True
BOARD_SIZE = 5
BOARD_SHAPE = BOARD_SIZE, BOARD_SIZE
BOARD_PLACES = tuple(product(range(BOARD_SIZE), repeat=2))
DIRECTIONS = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0))


class Position(NamedTuple):
    y: int
    x: int

    def __array__(self) -> npt.NDArray:
        array = np.zeros(shape=BOARD_SHAPE, dtype=np.int_)
        array[self] = 1
        return array
    
    def is_adjacent(self, other: 'Position'):
        return self != other and -1 <= self.x - other.x <= 1 and -1 <= self.y - other.y <= 1
    
    def __add__(self, other: object):
        return Position(self.y + other[0], self.x + other[1])


@dataclass(frozen=True, eq=False)
class Player:
    worker_0: Position
    worker_1: Position
    marker: str = ''

    @property
    def workers(self) -> Tuple[Position]:
        return self.worker_0, self.worker_1
    
    def move_worker(self, worker: Position, destination: Position) -> 'Player':
        worker_0 = destination if self.worker_0 == worker else self.worker_0
        worker_1 = destination if self.worker_1 == worker else self.worker_1
        return Player(worker_0, worker_1, self.marker)

    def __key(self):
        # Somehow np.ndarray.tostring() returns bytes not a string...
        return self.worker_0, self.worker_1

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Santorini) and self.__key() == __o.__key()
    
    def __hash__(self) -> int:
        return hash(self.__key())


class Action(NamedTuple):
    worker: Position
    destination: Position
    build: Position

    def as_tuple(self, game: 'Santorini'):
        return (
            game.current_player.workers.index(self.worker),
            self.destination.y,
            self.destination.x,
            self.build.y,
            self.build.x,
        )


@dataclass(frozen=True, eq=False)
class Santorini:
    player_0: Player
    player_1: Player

    # This is the board view, it should only be used for reading from the board, it is not immutable so be carefull `Santorini.board` will give a safe copy of the board.
    _board: npt.NDArray = field(default_factory=lambda: np.zeros(shape=BOARD_SHAPE, dtype=np.int_))
    
    # The current turn
    turn: int = 0

    @property
    def board(self) -> npt.NDArray:
        return self._board.copy()

    @property
    def players(self):
        return self.player_0, self.player_1
    
    @property
    def workers(self):
        return self.player_0.workers + self.player_1.workers
    
    @property
    def current_player_id(self):
        return self.turn % 2

    @property
    def current_player(self):
        return self.players[self.current_player_id]
    
    @property
    def non_current_player(self):
        return self.players[(self.turn + 1) % 2]
    
    def is_occupied(self, pos: Position):
        return self._board[pos] == 4 or pos in self.workers
    
    def is_on_board(self, pos: Position):
        return all(0 <= axis < BOARD_SIZE for axis in pos)

    def is_winning_action(self, action: Action):
        return self._board[action.worker] == 2 and self._board[action.destination] == 3

    def has_legal_action(self):
        try:
            next(self.get_legal_actions())
            return True
        except StopIteration:
            return False

    def is_legal_action(self, action: Action, safe=SAFE_MODE):
        if safe:
            if not action.worker in self.current_player.workers:
                # The selected worker does not exist for the current player.
                return False
            
            if not action.worker.is_adjacent(action.destination) or not action.destination.is_adjacent(action.build):
                # The selected move destination or build location is not within reach
                return False
            
        if not self.is_on_board(action.destination) or self.is_occupied(action.destination) or not self._board[action.worker] >= self._board[action.destination] - 1:
            # It is not legal to move to the selected position
            return False
        
        if not self.is_on_board(action.build) or (self.is_occupied(action.build) and action.worker != action.build):
            # It is not legal to build on the selected position
            return False
        
        return True
    
    def get_legal_actions(self):
        for worker in self.current_player.workers:
            for move_direction in DIRECTIONS:
                for build_direction in DIRECTIONS:
                    action = Action(
                        worker,
                        worker + move_direction,
                        worker + move_direction + build_direction
                    )

                    if self.is_legal_action(action, safe=False):
                        yield action
    
    def apply_legal_action(self, action):
        board = self.board
        board[action.build] += 1

        return Santorini(
            self.player_0.move_worker(action.worker, action.destination) if self.current_player_id == 0 else self.player_0, 
            self.player_1.move_worker(action.worker, action.destination) if self.current_player_id == 1 else self.player_1, 
            board, 
            self.turn + 1
        )

    def __key(self):
        # Somehow np.ndarray.tostring() returns bytes not a string...
        return self.player_0, self.player_1, self._board.tostring(), self.current_player_id

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Santorini) and self.__key() == __o.__key()
    
    def __hash__(self) -> int:
        return hash(self.__key())


    def render(self):
        from visualize import render_plotly
        fig = render_plotly((self, None))
        fig.show()

    @staticmethod
    def random_init(markers: Tuple[str] = ('0', '1'), random_board: bool = False):
        positions = [Position(*place) for place in random.sample(BOARD_PLACES, 2 * 2)]
        p0, p1 = Player(*positions[0:2], marker=markers[0]), Player(*positions[2:4], marker=markers[1])

        if random_board:
            board = np.random.randint(0, 4, size=BOARD_SHAPE)
        else:
            board = np.zeros(shape=BOARD_SHAPE, dtype=np.int_)
        
        return Santorini(p0, p1, board)


if __name__ == '__main__':
    print(Santorini.random_init(random_board=True))
