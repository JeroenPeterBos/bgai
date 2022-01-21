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
    
    def move(self, source: 'Position', destination: 'Position') -> 'Position':
        return destination if self == source else self

    def is_adjacent(self, other: 'Position'):
        return self != other and -1 <= self.x - other.x <= 1 and -1 <= self.y - other.y <= 1
    
    def __add__(self, other: object):
        return Position(self.y + other[0], self.x + other[1])


class Player(NamedTuple):
    worker_0: Position
    worker_1: Position

    @property
    def workers(self) -> Tuple[Position]:
        return self.worker_0, self.worker_1
    
    def move_worker(self, worker: Position, destination: Position) -> 'Player':
        return Player(self.worker_0.move(worker, destination), self.worker_1.move(worker, destination))


class Action(NamedTuple):
    worker: Position
    destination: Position
    build: Position


@dataclass(frozen=True, eq=False)
class Santorini:
    player_0: Player
    player_1: Player

    # This is the board view, it should only be used for reading from the board, it is not immutable so be carefull `Santorini.board` will give a safe copy of the board.
    _board: npt.NDArray = field(default_factory=lambda: np.zeros(shape=BOARD_SHAPE, dtype=np.int_))

    @property
    def board(self) -> npt.NDArray:
        return self._board.copy()

    @property
    def players(self):
        return self.player_0, self.player_1
    
    @property
    def workers(self):
        return self.player_0.workers + self.player_1.workers
    
    def worker_array(self) -> npt.NDArray:
        pieces = -np.ones(shape=BOARD_SHAPE, dtype=np.int_)
        
        for i, player in enumerate(self.players):
            for worker in player.workers:
                pieces[worker] = i
        
        return pieces
 
    def __key(self):
        # Somehow np.ndarray.tostring() returns bytes not a string...
        return self.player_0, self.player_1, self._board.tostring()

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Santorini) and self.__key() == __o.__key()
    
    def __hash__(self) -> int:
        return hash(self.__key())

    def __array__(self):
        return np.stack((
            self.player_0.worker_0, 
            self.player_0.worker_1,
            self.board,
            self.player_1.worker_0,
            self.player_1.worker_1
        ))

    def is_occupied(self, pos: Position):
        return self._board[pos] == 4 or pos in self.workers
    
    def is_on_board(self, pos: Position):
        return all(0 <= direction < BOARD_SIZE for direction in pos)

    def is_valid_move_action(self, worker: Position, destination: Position):
        if SAFE_MODE and not worker.is_adjacent(destination):
            return False

        return self.is_on_board(destination) and self._board[worker] >= self._board[destination] - 1 and not self.is_occupied(destination)

    def is_valid_build_action(self, worker: Position, build: Position):
        if SAFE_MODE and not worker.is_adjacent(build):
            return False
        
        return self.is_on_board(build) and not self.is_occupied(build)

    def is_valid_action(self, action: Action):
        return self.is_valid_move_action(action.worker, action.destination) and self.is_valid_build_action(action.worker, action.build)
    
    def is_winning_move(self, worker: Position, move: Position):
        return self._board[worker] == 2 and self._board[move] == 3
    
    def can_move(self, player: int):
        return any(self.is_valid_move_action(worker, worker + d) for worker in self.players[player].workers for d in DIRECTIONS)
    
    def apply_build_action(self, pos: Position):
        board = self.board
        board[pos] += 1
        return Santorini(self.player_0, self.player_1, board)
    
    def apply_move_action(self, worker: Position, move: Position):
        return Santorini(self.player_0.move_worker(worker, move), self.player_1.move_worker(worker, move), self.board)
    
    def apply_action(self, action: Action):
        return self.apply_move_action(action.worker, action.destination).apply_build_action(action.build)

    @staticmethod
    def random_init(random_board: bool = False):
        positions = [Position(*place) for place in random.sample(BOARD_PLACES, 2 * 2)]
        p0, p1 = Player(*positions[0:2]), Player(*positions[2:4])

        if random_board:
            board = np.random.randint(0, 4, size=BOARD_SHAPE)
        else:
            board = np.zeros(shape=BOARD_SHAPE, dtype=np.int_)
        
        return Santorini(p0, p1, board)


if __name__ == '__main__':
    print(Santorini.random_init(random_board=True))
