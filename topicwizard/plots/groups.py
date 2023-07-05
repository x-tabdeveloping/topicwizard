import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud


def group_topics_barchart(top_topics: pd.DataFrame):
    """Plots topic importances for currently selected group."""
    top_topics = top_topics.sort_values("importance", ascending=True)
    topic_word_trace = go.Bar(
        name="Estimated importance in group",
        y=top_topics.topic,
        x=top_topics.importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="#E03131",
    )
    overall_word_trace = go.Bar(
        name="Overall importance",
        y=top_topics.topic,
        x=top_topics.overall_importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        marker_color="rgb(168,162,158)",
        textposition="outside",
        texttemplate=top_topics.topic,
    )
    fig = go.Figure(data=[overall_word_trace, topic_word_trace])
    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="white",
        hovermode=False,
        uniformtext=dict(
            minsize=10,
            mode="show",
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.6)",
        ),
        margin=dict(l=0, r=0, b=18, t=0, pad=0),
    )
    fig.update_xaxes(
        range=[0, top_topics.overall_importance.max() * 1.3],
        showticklabels=False,
    )
    fig.update_yaxes(ticks="", showticklabels=False)
    fig.update_xaxes(
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
    )
    return fig


def wordcloud(top_words: pd.DataFrame) -> go.Figure:
    """Plots most relevant words for current topic as a worcloud."""
    top_dict = {
        word: importance
        for word, importance in zip(top_words.word, top_words.importance)
    }
    cloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        colormap="gnuplot",
        scale=4,
    ).generate_from_frequencies(top_dict)
    image = cloud.to_image()
    image = image.resize((1600, 1600), resample=Image.ANTIALIAS)
    fig = px.imshow(image)
    fig.update_layout(
        dragmode="pan",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    fig.update_yaxes(
        showticklabels=False,
        gridcolor="white",
        linecolor="white",
        zerolinecolor="white",
    )
    fig.update_xaxes(
        showticklabels=False,
        gridcolor="white",
        linecolor="white",
        zerolinecolor="white",
    )
    fig.update_traces(hovertemplate="", hoverinfo="none")
    return fig
