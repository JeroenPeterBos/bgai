import enum
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


def game_as_marker_array(game: Santorini, markers: Tuple[str] = None):
    if markers == None:
        markers = ('0', '1')

    board = np.empty(shape=BOARD_SHAPE, dtype=np.str_)

    for player, marker in zip(game.players, markers):
        for worker in player.workers:
            board[worker] = marker
    
    return board


def render_plotly(timesteps: List[Tuple[Santorini, Tuple[str]]]):
    if isinstance(timesteps, List):
        init_game, init_markers = timesteps[0]
        fig = px.imshow(
            np.stack(tuple(game.board for game, _ in timesteps)), 
            aspect="equal",
            color_continuous_scale=BLOCK_COLORS,
            zmin=0,
            zmax=4,
            animation_frame=0
        )
        
        for i, (game, markers) in enumerate(timesteps):
            fig.frames[i].data[0].text = game_as_marker_array(game, markers)
    else:
        init_game, init_markers = timesteps[0]
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
        text=game_as_marker_array(init_game, init_markers),
        textfont_size=42
    )
    
    return fig


def render_plotly_from_history(game: Santorini, history: Iterable[Action], markers: Tuple[str]):
    timesteps = [(game, markers)]

    for action in history:
        game = game.apply_legal_action(action)
        markers = markers[::-1]
        timesteps.append((game, markers))
    
    return render_plotly(timesteps)

if __name__ == '__main__':
    fig = render_plotly(
        [
            (Santorini.random_init(random_board=True), None),
            (Santorini.random_init(random_board=True), None),
            (Santorini.random_init(random_board=True), None),
        ]
    )

    fig.show()
