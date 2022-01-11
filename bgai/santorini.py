from dataclasses import dataclass, astuple
from collections import namedtuple
import enum
from itertools import combinations, product
from typing import Union
import random

import numpy as np
from numpy.typing import ArrayLike


BOARD_SIZE = 5

Position = namedtuple("Position", ["x", "y"])

@dataclass(frozen=True)
class Worker:
    pos: Position

    def as_array(self):
        board = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int_)
        board[self.pos.x, self.pos.y] = 1
        return board


@dataclass(frozen=True)
class Player:
    worker_0: Worker
    worker_1: Worker

    @property
    def workers(self):
        return (self.worker_0, self.worker_1)
    
    def as_array(self):
        return np.stack((worker.as_array() for worker in self.workers), axis=0)
    
    def move_worker(self, worker: Position, move: Position):
        return Player(
            Worker(move) if self.worker_0.pos == worker else self.worker_0,
            Worker(move) if self.worker_1.pos == worker else self.worker_1,
        )


@dataclass(frozen=True)
class Action:
    worker: Position
    move: Position
    build: Position


@dataclass(frozen=True)
class Santorini:
    player_0: Player
    player_1: Player
    board: ArrayLike = np.zeros(shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int_)

    def __post_init__(self):
        self.board.flags.writeable = False

        for a, b in combinations(self.workers, 2):
            if a.pos == b.pos:
                raise ValueError(f"Two workers have the same start position {a.pos}")
    
    @property
    def pieces(self):
        pieces = -np.ones(shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int_)
        
        for i, player in enumerate(self.players):
            for worker in player.workers:
                pieces[worker.pos] = i
        
        return pieces


    @property
    def players(self):
        return (self.player_0, self.player_1)
    
    @property
    def workers(self):
        return tuple(worker for player in self.players for worker in player.workers)
    
    def as_array(self):
        return np.concatenate((
            self.player_0.as_array(),
            self.board.reshape(1, BOARD_SIZE, BOARD_SIZE),
            self.player_1.as_array()
        ), axis=0)

    def is_occupied(self, pos: Position):
        return self.board[pos] == 4 or any(pos == worker.pos for worker in self.workers)
    
    def is_unoccupied(self, pos: Position):
        return not self.is_occupied(pos)

    def is_valid_move_action(self, worker: Position, move: Position):
        return 0 <= move.x < BOARD_SIZE and 0 <= move.y < BOARD_SIZE and self.board[worker] >= self.board[move] - 1 and self.is_unoccupied(move)

    def is_valid_build_action(self, worker: Position, build: Position):
        return 0 <= build.x < BOARD_SIZE and 0 <= build.y < BOARD_SIZE and self.is_unoccupied(build) or worker == build

    def is_valid_action(self, action: Action):
        return self.is_valid_move_action(action.worker, action.move) and self.is_valid_build_action(action.worker, action.build)
    
    def is_winning_move(self, worker: Position, move: Position):
        return self.board[worker] == 2 and self.board[move] == 3
    
    def can_move(self, player: int):
        return any(self.is_valid_move_action(worker.pos, Position(worker.pos.x + i, worker.pos.y + j)) for worker in self.players[player].workers for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0))
    
    def apply_build_action(self, pos: Position):
        board = np.array(self.board)
        board[pos] += 1
        return Santorini(self.player_0, self.player_1, board)
    
    def apply_move_action(self, worker: Position, move: Position):
        return Santorini(self.player_0.move_worker(worker, move), self.player_1.move_worker(worker, move), self.board)
    
    def apply_action(self, action: Action):
        return self.apply_move_action(action.worker, action.move).apply_build_action(action.build)

    @staticmethod
    def random_board_init(random_board: bool = False):
        positions = list(map(Position._make, product(range(BOARD_SIZE), repeat=2)))
        random.shuffle(positions)
        
        p0 = Player(Worker(positions[0]), Worker(positions[1]))
        p1 = Player(Worker(positions[2]), Worker(positions[3]))

        if random_board:
            return Santorini(p0, p1, np.random.randint(0, 4, size=(BOARD_SIZE, BOARD_SIZE)))
        else:
            return Santorini(p0, p1)


if __name__ == '__main__':
    print(Santorini.random_board_init(random_board=True))
