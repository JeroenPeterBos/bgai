import random
import click
import logging

log = logging.getLogger(__name__)

from bgai.visualize import render_plotly_from_history
from bgai.santorini import DIRECTIONS, Action, Position, Santorini


class BasePlayer:
    player_type = "base"

    def __init__(self, player_id: int):
        self._player_id = player_id
    
    def __str__(self):
        return f"Player '{self.player_type}' with id {self._player_id}"


class InputPlayer(BasePlayer):
    player_type = "input"

    def get_action(game: Santorini):

        # Retrieve valid action from input()

        return 


class RandomPlayer(BasePlayer):
    player_type = "random"

    def get_random_action(self, game: Santorini):
        workers = game.workers(self._player_id)

        for worker in random.sample(workers, len(workers)):
            for move_direction in random.sample(DIRECTIONS, len(DIRECTIONS)):
                destination = worker + move_direction

                if game.is_valid_move_action(worker, destination):
                    moved_game = game.apply_move_action(worker, destination)

                    for build_direction in random.sample(DIRECTIONS, len(DIRECTIONS)):
                        build = destination + build_direction

                        if moved_game.is_valid_build_action(destination, build):
                            return Action(worker, destination, build)
        
        raise ValueError(f"The game did not finish, but no valid move could be found in game {game} for player {self._player_id}.")

    def get_action(self, game: Santorini):
        return self.get_random_action(game)


class ClimberPlayer(RandomPlayer):
    player_type = "climber"

    def get_action(self, game: Santorini):
        for worker in game.workers(self._player_id):
            for move_direction in DIRECTIONS:
                destination = worker + move_direction

                if game.is_valid_move_action(worker, destination):
                    moved_game = game.apply_move_action(worker, destination)

                    if game._board[worker] < moved_game._board[destination]:

                        build_options = []
                        for build_direction in DIRECTIONS:
                            build = destination + build_direction

                            if moved_game.is_valid_build_action(destination, build):
                                worker_height = moved_game._board[destination]
                                build_current_height = moved_game._board[build]
                                difference = worker_height - build_current_height
                                # Pick the heighest build spot, unless it is already 3 high

                                if difference >= 0:
                                    score = difference
                                else:
                                    score = -difference + 2
                                build_options.append((build, score))
                            
                        if len(build_options) > 0:
                            build, _ = min(build_options, key=lambda x: x[1])
                            return Action(worker, destination, build)
        
        return self.get_random_action(game)


PLAYER_TYPES = { c.player_type: c for c in (InputPlayer, RandomPlayer, ClimberPlayer)}


def play_game(game, players, current_player):
    is_won = False
    history = []

    log.info(f"The players are {players[0]} and {players[1]}")
    log.info(f"The initial game state is: {game}.")

    while not is_won:
        player = players[current_player]
        action = player.get_action(game)
        log.info(f"Turn {len(history)} | Player {player} plays {action}")

        history.append(action)

        new_game = game.apply_action(action)

        current_player = (current_player + 1) % 2
        is_won = game.is_winning_move(action.worker, action.destination) or not new_game.can_move(current_player)

        game = new_game
    
    log.info(f"The game is won by player {player}.")
    
    return tuple(history)


@click.command()
@click.argument("player_a", type=click.Choice(PLAYER_TYPES.keys(), case_sensitive=False))
@click.argument("player_b", type=click.Choice(PLAYER_TYPES.keys(), case_sensitive=False))
@click.option("--html", default=None, type=click.Path(resolve_path=True))
def cli(player_a, player_b, html):
    log.info(f"Called the cli with arguments: {player_a}, {player_b}, {html}")

    players = PLAYER_TYPES[player_a](0), PLAYER_TYPES[player_b](1)
    game = Santorini.random_init()
    start_player = random.randint(0, 1)

    log.info("Playing the game...")
    history = play_game(game, players, start_player)

    if html is not None:
        log.info("Writing the game to an html file")
        fig = render_plotly_from_history(game, history)
        fig.write_html(html)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cli()