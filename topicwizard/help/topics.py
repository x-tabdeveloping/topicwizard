import dash_mantine_components as dmc
from dash_extensions.enrich import dcc, html

TOPIC_MAP = [
    dmc.Title("Topic Map", order=3),
    dmc.Text(
        "On the left side of this page you will find the topic map. It will help you examine topic importances and intertopic distances.",
    ),
    dmc.Text(
        "To select a topic to investigate, click on it on the map, or cycle to it with the arrow buttons on top."
    ),
    dmc.Text("You can also rename topics using the bar at the top."),
    dmc.Title("How are distances calculated?", order=4),
    dmc.Text(
        """
        Distances are based on the topic-term matrices.
        They are projected into 2-dimensional space using UMAP.
        """
    ),
    dmc.Title("How are sizes calculated?", order=4),
    dmc.Text(
        "Topic importances are initially calculated from the document-topic matrix like this:"
    ),
    dmc.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(
                            dcc.Markdown(
                                "$t_i = \\sum_j d_{j_i} \\cdot |d_j|$", mathjax=True
                            )
                        ),
                        html.Td("Importance of topic i."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$d_{j_i}$", mathjax=True)),
                        html.Td("Importance of topic i for document j."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$|d_j|$", mathjax=True)),
                        html.Td("Length of document j."),
                    ]
                ),
            ],
        )
    ),
    dmc.Text(
        """
        This may result in negative values, which we of course can't display on the graph.
        Therefore we MinMax normalize the values and add a smoothing of + 1.
        Sizes on the graph are proportional to this value.
        """
    ),
]

TOPIC_WORDS = [
    dmc.Title("Important Words", order=3),
    dmc.Text(
        """
        On the right side of the page you will find a bar plot displaying importances
        of words for the selected topic and their overall importance,
        along with a wordcloud displaying the most relevant words for the selected topic.
        """
    ),
    dmc.Title("How are importances calculated?", order=4),
    dmc.Text(
        """
    Importance in a single topic can be obtained by multiplying the topic's importance with values from the topic-term matrix.
    """,
    ),
    dmc.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$w_{k_i} = \\phi_{k_i}$", mathjax=True)),
                        html.Td("Importance of word k for topic i."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            dcc.Markdown("$w_{k} = \\sum_i w_{k_i}$", mathjax=True)
                        ),
                        html.Td("Summed up word importance over all topics."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$\\phi$", mathjax=True)),
                        html.Td("Topic-term matrix from the model."),
                    ]
                ),
            ],
        )
    ),
]
