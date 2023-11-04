from string import Template
from typing import Tuple

import numpy as np
from dash_extensions.enrich import DashBlueprint, Input, Output, State, dcc
from scipy.stats import zscore

import topicwizard.plots.words as plots


def create_word_map(
    word_positions: Tuple[np.ndarray, np.ndarray],
    word_frequencies: np.ndarray,
    vocab: np.ndarray,
    dominant_topic: np.ndarray,
    topic_colors: np.ndarray,
):
    x, y = word_positions

    word_map = DashBlueprint()

    word_map.layout = dcc.Graph(
        id="word_map",
        responsive=True,
        # config=dict(scrollZoom=True),
        # animate=True,
        figure=plots.word_map(
            x=x,
            y=y,
            word_frequencies=word_frequencies,
            vocab=vocab,
            dominant_topic=dominant_topic,
            topic_colors=topic_colors,
        ),
        className="flex-1",
    )
    z_values = zscore(word_frequencies)
    highest = np.arange(len(vocab))[z_values > 2.0]
    highest = highest[np.argsort(-z_values[highest])[:40]]
    highest_values_str = ", ".join([str(int(h)) for h in highest])
    highest_text = f"[{highest_values_str}]"

    word_map.clientside_callback(
        Template(
            """
        function(selectedWords, associatedWords, currentPlot, vocab) {
            opacity = 0.4
            if (selectedWords === undefined || selectedWords.length == 0) {
                opacity = 0.5
                selectedWords = $highest
            }
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
                textColor[index] = 'black';
                textSize[index] = 16;
            })
            associatedWords.forEach(index => {
                text[index] = vocab[index];
                textSize[index] = 10;
                textColor[index] = 'rgba(40,40,40,0.6)';
                colors[index] = '#89dcc3';
            })
            const textFont = {'color': textColor, 'size': textSize};
            // const marker = {...trace.marker, 'color': colors};
            const marker = {...trace.marker, 'opacity': opacity}
            const newTrace = {...trace, 'marker': marker, 'text': text, 'textfont': textFont};
            const newFigure = {...currentPlot, 'data': [newTrace]};
            return newFigure;
        }
        """
        ).substitute(highest=highest_text),
        Output("word_map", "figure"),
        Input("selected_words", "data"),
        Input("associated_words", "data"),
        State("word_map", "figure"),
        State("vocab", "data"),
        prevent_initial_call=True,
    )
    return word_map
