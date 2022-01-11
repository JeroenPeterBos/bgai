import random
import click
import logging

log = logging.getLogger(__name__)

from bgai.visualize import render_plotly_from_history
from bgai.santorini import Action, Position, Santorini


DIRECTIONS = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0))


class BasePlayer:
    def __init__(self, player_id: int):
        self._player_id = player_id


class InputPlayer(BasePlayer):
    def get_action(game: Santorini, player_id: int):

        # Retrieve valid action from input()

        return 


class RandomPlayer(BasePlayer):
    def get_action(self, game: Santorini):
        workers = list(game.players[self._player_id].workers)
        random.shuffle(workers)

        for worker in workers:
            moves = list(DIRECTIONS)
            random.shuffle(moves)

            for i, j in moves:
                move = Position(worker.pos.x + i, worker.pos.y + j)

                if game.is_valid_move_action(worker.pos, move):
                    moved_game = game.apply_move_action(worker.pos, move)

                    builds = list(DIRECTIONS)
                    random.shuffle(builds)

                    for k, l in builds:
                        build = Position(move.x + k, move.y + l)

                        if moved_game.is_valid_build_action(move, build):
                            return Action(worker.pos, move, build)
        
        raise ValueError(f"The game did not finish, but no valid move could be found in game {game} for player {self._player_id}.")


def get_player(player_type: str, player_id: int):
    player_type = player_type.lower()
    if player_type == 'input':
        return InputPlayer(player_id)
    elif player_type == 'random':
        return RandomPlayer(player_id)


def play_game(game, players, current_player):
    is_won = False
    history = []

    while not is_won:
        action = players[current_player].get_action(game)
        log.info(f"Turn {len(history)} | Player {current_player} plays {action}")

        history.append(action)

        new_game = game.apply_action(action)

        current_player = (current_player + 1) % 2
        is_won = game.is_winning_move(action.worker, action.move) or not new_game.can_move(current_player)

        game = new_game
    
    log.info(f"The game is won by player {current_player}.")
    
    return tuple(history)


@click.command()
@click.argument("player_a", type=click.Choice(["input", "random"], case_sensitive=False))
@click.argument("player_b", type=click.Choice(["input", "random"], case_sensitive=False))
@click.option("--html", default=None, type=click.Path(resolve_path=True))
def cli(player_a, player_b, html):
    log.info(f"Called the cli with arguments: {player_a}, {player_b}, {html}")

    players = (get_player(player_a, 0), get_player(player_b, 1))
    game = Santorini.random_board_init()
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