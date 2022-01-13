import random
from typing import NamedTuple, Tuple
from itertools import combinations, product
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


BOARD_SIZE = 5
BOARD_SHAPE = BOARD_SIZE, BOARD_SIZE
BOARD_PLACES = tuple(product(range(BOARD_SIZE), repeat=2))
DIRECTIONS = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0))


class Position(NamedTuple):
    y: int
    x: int

    def as_array(self) -> npt.NDArray:
        array = np.zeros(shape=BOARD_SHAPE, dtype=np.int_)
        array[self] = 1
        return array
    
    def move(self, source: 'Position', destination: 'Position') -> 'Position':
        return destination if self == source else self
    
    def __add__(self, other: object):
        return Position(self.y + other[0], self.x + other[1])


class Player(NamedTuple):
    worker_0: Position
    worker_1: Position

    @property
    def workers(self) -> Tuple[Position]:
        return self.worker_0, self.worker_1
    
    def as_array(self) -> npt.NDArray:
        return np.stack((worker.as_array() for worker in self.workers), axis=0)
    
    def move_worker(self, worker: Position, destination: Position) -> 'Player':
        return Player(self.worker_0.move(worker, destination), self.worker_1.move(worker, destination))


class Action(NamedTuple):
    worker: Position
    destination: Position
    build: Position


class Santorini(NamedTuple):
    player_0: Player
    player_1: Player

    # This is the board view, it should only be used for reading from the board, it is not immutable so be carefull `Santorini.board` will give a safe copy of the board.
    board_readonly: npt.NDArray

    @property
    def board(self) -> npt.NDArray:
        return self.board_readonly.copy()

    @property
    def pieces(self) -> npt.NDArray:
        pieces = -np.ones(shape=BOARD_SHAPE, dtype=np.int_)
        
        for i, player in enumerate(self.players):
            for worker in player.workers:
                pieces[worker] = i
        
        return pieces

    @property
    def players(self):
        return self.player_0, self.player_1
    
    def workers(self, player_id=None):
        if player_id is not None:
            return self.players[player_id].workers
        else:
            return self.player_0.workers + self.player_1.workers
    
    def as_array(self):
        return np.concatenate((
            self.player_0.as_array(),
            self.board.reshape(1, BOARD_SIZE, BOARD_SIZE),
            self.player_1.as_array()
        ), axis=0)

    def is_occupied(self, pos: Position):
        return self.board_readonly[pos] == 4 or pos in self.workers()
    
    def is_unoccupied(self, pos: Position):
        return not self.is_occupied(pos)
    
    def is_on_board(self, pos: Position):
        return all(0 <= direction < BOARD_SIZE for direction in pos)

    def is_valid_move_action(self, worker: Position, destination: Position):
        return self.is_on_board(destination) and self.board_readonly[worker] >= self.board_readonly[destination] - 1 and self.is_unoccupied(destination)

    def is_valid_build_action(self, worker: Position, build: Position):
        return self.is_on_board(build) and self.is_unoccupied(build)

    def is_valid_action(self, action: Action):
        return self.is_valid_move_action(action.worker, action.destination) and self.is_valid_build_action(action.worker, action.build)
    
    def is_winning_move(self, worker: Position, move: Position):
        return self.board_readonly[worker] == 2 and self.board_readonly[move] == 3
    
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
