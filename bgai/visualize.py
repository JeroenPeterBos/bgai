import enum
from typing import Iterable, List, Union
from os.path import expanduser

import numpy as np
from numpy.typing import ArrayLike
import plotly.express as px
import plotly.graph_objects as go

from bgai.santorini import Action, Santorini, BOARD_SIZE


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


def render_plotly(games: Santorini):
    pieces_format = np.vectorize(lambda x: '' if x < 0 else str(x))

    if isinstance(games, Iterable):
        game = games[0]
        fig = px.imshow(
            np.stack(tuple(game.board for game in games)), 
            aspect="equal",
            color_continuous_scale=BLOCK_COLORS,
            zmin=0,
            zmax=4,
            animation_frame=0
        )
        
        for i, game in enumerate(games):
            fig.frames[i].data[0].text = pieces_format(game.pieces)
    else:
        game = games
        fig = px.imshow(
            game.board, 
            aspect="equal",
            color_continuous_scale=BLOCK_COLORS,
            zmin=0,
            zmax=4,
        )
 
    fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)", paper_bgcolor="rgba(0, 0, 0, 0)")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(
        xgap=10, 
        ygap=10,
        hovertemplate='x:         %{x}<br>y:         %{y}<br>height: %{z}<br>player: %{text}<extra></extra>',
        selector=dict(type='heatmap'),
        texttemplate="<b>%{text}</b>",
        text=pieces_format(game.pieces),
        textfont_size=42
    )
    
    return fig


def render_plotly_from_history(game: Santorini, history: Iterable[Action]):
    games = [game]

    for action in history:
        game = game.apply_action(action)
        games.append(game)
    
    return render_plotly(games)

if __name__ == '__main__':
    fig = render_plotly(
        [
            Santorini.random_board_init(random_board=True),
            Santorini.random_board_init(random_board=True),
            Santorini.random_board_init(random_board=True),
        ]
    )

    fig.show()
