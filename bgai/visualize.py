import enum
from sqlite3 import Timestamp
from typing import Iterable, List, Tuple, Union
from os.path import expanduser

import numpy as np
from numpy.typing import ArrayLike
import plotly.express as px
import plotly.graph_objects as go

from bgai.santorini import BOARD_SHAPE, Action, Santorini


def _discrete_color_scale(*colors):
    return [((elem + i) / len(colors), color) for elem, color in enumerate(colors) for i in range(2)]

BLOCK_COLORS = _discrete_color_scale(
    # "#656CE0",
    # "#D14C34",
    "#3EC08D",
    # "#008A5A",
    "#e9e7e8",
    "#ACC3D5",
    "#6e9ec1",
    "#013485"
)


def game_as_marker_array(game: Santorini):
    board = np.empty(shape=BOARD_SHAPE, dtype=np.str_)

    for player in game.players:
        for worker in player.workers:
            board[worker] = player.marker
    
    return board


def render_plotly(timesteps: List[Santorini]):
    if isinstance(timesteps, list):
        init_game = timesteps[0]
        fig = px.imshow(
            np.stack(tuple(game.board for game in timesteps)), 
            aspect="equal",
            color_continuous_scale=BLOCK_COLORS,
            zmin=0,
            zmax=4,
            animation_frame=0
        )
        
        for i, game in enumerate(timesteps):
            fig.frames[i].data[0].text = game_as_marker_array(game)
    else:
        init_game = timesteps
        fig = px.imshow(
            init_game.board, 
            aspect="equal",
            color_continuous_scale=BLOCK_COLORS,
            zmin=0,
            zmax=4,
        )
 
    fig.update_layout(plot_bgcolor="#669C82", paper_bgcolor="rgba(0, 0, 0, 0)", xaxis_gridcolor="#669C82", yaxis_gridcolor="#669C82")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(
        xgap=10, 
        ygap=10,
        hovertemplate='y:         %{y}<br>x:         %{x}<br>height: %{z}<br>player: %{text}<extra></extra>',
        selector=dict(type='heatmap'),
        texttemplate="<b>%{text}</b>",
        text=game_as_marker_array(init_game),
        textfont_size=42
    )
    
    return fig


def render_history(game: Santorini, history: Iterable[Action]):
    timesteps = [game]

    for action in history:
        game = game.apply_legal_action(action)
        timesteps.append(game)
    
    return render_plotly(timesteps)

def render_path(path, show=True):
    timesteps = []

    for node in path:
        timesteps.append(node.game)
    
    fig = render_plotly(timesteps)

    if show:
        fig.show()
    
    return fig


if __name__ == '__main__':
    fig = render_plotly(
        [
            (Santorini.random_init(random_board=True), None),
            (Santorini.random_init(random_board=True), None),
            (Santorini.random_init(random_board=True), None),
        ]
    )

    fig.show()
