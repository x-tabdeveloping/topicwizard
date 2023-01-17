from typing import Tuple

import numpy as np

from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
    Input,
    Output,
    State,
)
import topicwizard.plots.words as plots


def create_word_map(
    word_positions: Tuple[np.ndarray, np.ndarray],
    word_frequencies: np.ndarray,
    vocab: np.ndarray,
):
    x, y = word_positions

    word_map = DashBlueprint()

    word_map.layout = dcc.Graph(
        id="word_map",
        responsive=True,
        # config=dict(scrollZoom=True),
        # animate=True,
        figure=plots.word_map(
            x=x, y=y, word_frequencies=word_frequencies, vocab=vocab
        ),
        className="flex-1",
    )

    word_map.clientside_callback(
        """
        function(selectedWords, associatedWords, currentPlot, vocab) {
            if (!currentPlot || !vocab){
                return {'data': [], 'layout': {}};
            }
            const trigerred = window.dash_clientside.callback_context.triggered[0];
            const trigerredId = trigerred.prop_id.split(".")[0];
            const trace = currentPlot.data[0];
            const nVocab = vocab.length;
            const text = new Array(nVocab).fill('');
            const colors = new Array(nVocab).fill('#a8a29e');
            const textColor = new Array(nVocab).fill('black');
            const textSize = new Array(nVocab).fill(9);
            selectedWords.forEach(index => {
                text[index] = vocab[index];
                colors[index] = '#15AABF';
                textColor[index] = '#0B7285';
                textSize[index] = 22;
            })
            associatedWords.forEach(index => {
                text[index] = vocab[index];
                colors[index] = '#89dcc3';
            })
            const textFont = {'color': textColor, 'size': textSize};
            const marker = {...trace.marker, 'color': colors};
            const newTrace = {...trace, 'marker': marker, 'text': text, 'textfont': textFont};
            const newFigure = {...currentPlot, 'data': [newTrace]};
            return newFigure;
        }
        """,
        Output("word_map", "figure"),
        Input("selected_words", "data"),
        Input("associated_words", "data"),
        State("word_map", "figure"),
        State("vocab", "data"),
    )
    return word_map
