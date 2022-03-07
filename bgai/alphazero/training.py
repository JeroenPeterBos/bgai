from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Empty, Queue
import numpy as np
from datetime import date, datetime, timedelta
from bgai.alphazero.mcts import Node, mcts
from bgai.santorini import BOARD_SIZE, Santorini
import bgai.timer as timer

import logging
log = logging.getLogger(__name__)


class SantoriniTracker:
    POLICY_SHAPE = (2, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE)

    def __init__(self):
        self.states = []
        self.visits = []
        self.action_count = 0
    
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
            visits_array[action.as_tuple(game)] = child.visit_count / total_visits

        self.states.append(game_array)
        self.visits.append(visits_array)
        self.action_count += 1


class TrainerLink:
    def __init__(self, window_size: int, batch_size: int, fetch_min_wait: int):
        self._queue = Queue()
        self._window = []
        self._last_fetch = datetime.now()

        self.online = True
        self.window_size = window_size
        self.batch_size = batch_size
        self.fetch_min_wait = fetch_min_wait
    
    def publish_tracker(self, tracker: SantoriniTracker):
        self._queue.put(tracker)
    
    def move_trackers_from_queue_to_window(self):
        done = False
        while not done:
            maximum_wait = self.fetch_min_wait - (datetime.now() - self._last_fetch).total_seconds()
            try:
                if maximum_wait > 0:
                    tracker = self._queue.get(block=True, timeout=maximum_wait)
                else:
                    tracker = self._queue.get(block=False, timeout=None)

                if len(self._window) >= self.window_size:
                    self._window.pop(0)
                
                self._window.append(tracker)
            except Empty:
                done = True
                self._last_fetch = datetime.now()
    
    def sample_batch(self):
        total_actions = sum(tracker.action_count for tracker in self._window)
        tracker_selection = np.random.choice(
            self._window,
            size=self.batch_size,
            p=[len(tracker.action_count) / total_actions for tracker in self._window]
        )
        action_selection = [(tracker, np.random.randint(len(tracker.action_count))) for tracker in tracker_selection]

        # Implement sample creation (turn game into img etc)
        # return [ for tracker, action_index in tracker_selection]



def selfplay(link: TrainerLink):
    tracker = SantoriniTracker()
    game = Santorini.random_init()
    log.info("Entering selfplay")
    is_terminal = False

    turn_counter = 0
    while not is_terminal and link.online:
        turn_counter += 1
        log.info(f"Starting turn {turn_counter}")

        with timer.Timer():
            action, root = mcts(game)
        tracker.track_statistics(game, root)

        is_terminal = game.is_winning_action(action)
        game = game.apply_legal_action(action)

    return tracker


def runner(runner_id: int, link: TrainerLink):
    log.info(f"Runner {runner_id} is starting a new game.")

    tracker = selfplay(link)
    link.publish_tracker(tracker)


def trainer(link: TrainerLink):
    # TODO: Enter loop in which games will be pulled from the queue and sample batches will be used to train and publish the new network

    link.move_trackers_from_queue_to_window()
    log.info(f"Link window currently has size {len(link._window)}")


def main(threads: int, steps=5):
    log.info("Starting the training main function")
    link = TrainerLink(
        window_size=50,
        fetch_min_wait=1
    )

    if threads == 1:
        log.info("Running the runner trainer loops alternating on a single thread.")
        for step in range(steps):
            log.info(f"Step {step}")
            runner(runner_id=0, link=link)
            trainer(link)
    else:
        log.info(f"Creating ThreadPoolExecutor with {threads} threads")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(partial(runner, link=link), range(threads))

            for _ in range(steps):
                trainer(link)

            log.info("Disabling the trainer link to stop the runners.")
            link.online = False
        
    log.info("Finished")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(thread)-2d | %(name)25s:%(lineno)-3d | %(funcName)-30s | %(message)s")
    main(1)

    timings = np.array(timer.timings)
    log.info(f"The timer recorded an average time of {timings.mean():.3f} with a standard deviation of {timings.std():.3f}")
