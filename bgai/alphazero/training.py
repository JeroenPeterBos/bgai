from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Empty, Queue
import time
import numpy as np
from datetime import date, datetime, timedelta
from bgai.alphazero.mcts import Node, mcts
from bgai.santorini import BOARD_SIZE, Santorini

import logging
log = logging.getLogger(__name__)


class SantoriniTracker:
    POLICY_SHAPE = (2, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE)

    def __init__(self):
        self.states = []
        self.visits = []
    
    def track_statistics(self, game: Santorini, root: Node):
        game_array = np.stack((
            game.current_player.worker_0,
            game.current_player.worker_1,
            game.board,
            game.non_current_player.worker_0,
            game.non_current_player.worker_1
        ))

        total_visits = sum(map(lambda c: c.visit_count, root.children.values()))
        visits_array = np.zeros(shape=self.POLICY_SHAPE, dtype=np.int_)
        for action, child in root.children.items():
            visits_array[action.as_tuple()] = child.visit_count / total_visits

        self.states.append(game_array)
        self.visits.append(visits_array)


class TrainerLink:
    def __init__(self, window_size: int, fetch_min_wait: int):
        self._queue = Queue()
        self._window = []
        self._last_fetch = datetime.now()

        self.online = True
        self.window_size = window_size
        self.fetch_min_wait = fetch_min_wait
    
    def publish_tracker(self, tracker: SantoriniTracker):
        self._queue.put(tracker)
    
    def fetch_new_trackers(self):
        done = False
        while not done:
            maximum_wait = self.fetch_min_wait - (datetime.now() - self._last_fetch).total_seconds()
            try:
                if maximum_wait > 0:
                    log.info(f"Waiting for {maximum_wait} for games to get published.")
                    tracker = self._queue.get(block=True, timeout=maximum_wait)
                else:
                    tracker = self._queue.get(block=False, timeout=None)

                if len(self._window) >= self.window_size:
                    self._window.pop(0)
                
                self._window.append(tracker)
            except Empty:
                done = True
                self._last_fetch = datetime.now()
        

def runner(runner_id: int, link: TrainerLink):
    game = 0
    while link.online:
        game += 1
        log.info(f"Runner {runner_id} is starting game {game}.")

        tracker = selfplay(link)
        link.publish_tracker(tracker)


def selfplay(link: TrainerLink):
    tracker = SantoriniTracker()
    game = Santorini()
    is_terminal = False

    turn_counter = 0
    while not is_terminal and link.online:
        turn_counter += 1
        log.info(f"Starting turn {turn_counter}")

        action, root = mcts(game)
        game = game.apply_legal_action(action)
        tracker.track_statistics(game, root)

    return tracker


def main(threads: int = 10):
    log.info("Starting the training main function")
    link = TrainerLink(50, 3)

    log.info(f"Creating ThreadPoolExecutor with {threads} threads")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(partial(runner, link=link), range(threads))
        log.info("Created and mapped the threads to the runner function")
        
        # TODO: Enter loop in which games will be pulled from the queue and sample batches will be used to train and publish the new network

        for _ in range(120):
            log.info(f"Link window currently has size {len(link._window)}")

            time.sleep(1)
            
            link.fetch_new_trackers()

        log.info("Done, turning off the trainer link")
        link.online = False
    
    log.info("Finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(thread)-2d | %(name)25s:%(lineno)-3d | %(funcName)-30s | %(message)s")
    main()
