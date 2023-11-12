import dash_mantine_components as dmc
from dash_extensions.enrich import dcc, html

WORD_MAP = [
    dmc.Title("Word Map", order=3),
    dmc.Text(
        "On the left side of this page you will find the word map. It will help you examine the semantic structure of your corpus.",
    ),
    dmc.Text(
        """To select a word click on it on the map or search for it in the top bar.
        You can select multiple words at a time.
        This will also highlight the words most closely associated with the selected words.
        """
    ),
    dmc.Text("You can adjust the number of associations with the slider."),
    dmc.Title("How are positions calculated?", order=4),
    dmc.Text(
        "Embedded word representations are obtained by transposing the topic-term matrix from the topic model."
    ),
    dmc.Text("These representations are then projected into 2D space with UMAP."),
    dmc.Title("What do sizes correspond to?", order=4),
    dmc.Text(
        "Sizes are calculated proportional to the frequency of the words in the corpus."
    ),
    dmc.Title("How are words colored?", order=4),
    dmc.Text(
        "Colors are determined by the most important topic for a given word. Each topic has a different color."
    ),
    dmc.Text(
        "The most important topic is determined by taking the highest topic value in the topic-term matrix."
    ),
    dmc.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(
                            dcc.Markdown("$c_k = argmax_i \\phi_{i_k}$", mathjax=True)
                        ),
                        html.Td("Most important topic for word k"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$\\phi_{i_k}$", mathjax=True)),
                        html.Td("Importance of word k for topic i."),
                    ]
                ),
            ],
        )
    ),
    dmc.Title("How are associations obtained?", order=4),
    dmc.Text(
        "Associations are selected based on closest cosine distance between word representations."
    ),
]

TOPIC_IMPORTANCES = [
    dmc.Title("Topic Importances", order=3),
    dmc.Text(
        """
        On the right side of the page you will find a bar plot displaying topic importances for the selected words
        and their closest associations.
        """
    ),
    dmc.Title("How are topic importances calculated?", order=4),
    dmc.Text(
        "The importance of a topic for a word is the exact same as the importance of a word for a topic."
    ),
    dmc.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$t_{i_k} = \\phi_{i_k}$", mathjax=True)),
                        html.Td("Importance of topic i for word k."),
                    ]
                ),
            ],
        )
    ),
    dmc.Text("These values are then summed for all associated words."),
]
