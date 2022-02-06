from lib2to3.pytree import Base
import random
from typing import Tuple
import click
import logging
from bgai.player import PLAYER_TYPES, BasePlayer

log = logging.getLogger(__name__)

from bgai.visualize import render_plotly_from_history
from bgai.santorini import DIRECTIONS, Action, Position, Santorini


def play_game(game: Santorini, players: Tuple[BasePlayer]):
    is_won = False
    history = []
    current_player = 0

    log.info(f"The players are {players[0]} and {players[1]}")
    log.info(f"The initial game state is: {game}.")

    while not is_won:
        player = players[current_player]
        action = player.get_action(game)
        log.info(f"Turn {len(history)} | Player {player} plays {action}")

        history.append(action)

        new_game = game.apply_legal_action(action)

        current_player = (current_player + 1) % 2
        is_won = game.is_winning_action(action) or not new_game.has_legal_action()

        game = new_game
    
    log.info(f"The game is won by player {player}.")
    
    return tuple(history)


@click.command()
@click.argument("player_a", type=click.Choice(PLAYER_TYPES.keys(), case_sensitive=False))
@click.argument("player_b", type=click.Choice(PLAYER_TYPES.keys(), case_sensitive=False))
@click.option("--html", default=None, type=click.Path(resolve_path=True))
def cli(player_a, player_b, html):
    log.info(f"Called the cli with arguments: {player_a}, {player_b}, {html}")

    players = PLAYER_TYPES[player_a](0, 'O'), PLAYER_TYPES[player_b](1, 'X')
    game = Santorini.random_init()

    log.info("Playing the game...")
    history = play_game(game, players)

    if html is not None:
        log.info(f"Writing the game to {html}")
        fig = render_plotly_from_history(game, history, tuple(player._marker for player in players))
        fig.write_html(html)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cli()