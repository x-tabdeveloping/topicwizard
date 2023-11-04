import dash_mantine_components as dmc
from dash_extensions.enrich import dcc, html

DOCUMENT_MAP = [
    dmc.Title("Document Map", order=3),
    dmc.Text(
        "On the left side of this page you will find a map of documents in your corpus.",
    ),
    dmc.Text(
        """
        You can select and search for documents in the top bar or by clicking at them in the plot.
        """
    ),
    dmc.Title("How are positions calculated?", order=4),
    dmc.Text(
        """
        Document representations are obtained from the document-topic matrix.
        """
    ),
    dmc.Text("These representations are then projected into 2D space with UMAP."),
    dmc.Title("How are documents colored?", order=4),
    dmc.Text(
        "Colors are determined by the most important topic in e given document. Each topic has a different color."
    ),
    dmc.Text(
        "The most important topic is determined by taking the highest topic value in the document-topic matrix."
    ),
    dmc.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$c_j = argmax_i d_{j_i}$", mathjax=True)),
                        html.Td("Most important topic for document j"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$d_{j_i}$", mathjax=True)),
                        html.Td("Importance of topic i for document j."),
                    ]
                ),
            ],
        )
    ),
]

CONTENT = [
    dmc.Title("Content", order=3),
    dmc.Text(
        """
        In the upper-left corner you will find a component showing the content of the selected document.
        Words which are 'significant' for the most important topic in the document are highlighted in grey.
        """
    ),
    dmc.Title("Which words are highlighted?", order=4),
    dmc.Text(
        "Words which have a higher z-score in the most important topic than 2.0 are highlighted."
    ),
    dmc.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(
                            dcc.Markdown(
                                "$z_{k_i} = \\frac{\\phi_{i_k} - \\bar{\\phi_i}}{S_{\\phi_i}}$",
                                mathjax=True,
                            )
                        ),
                        html.Td("Z-score of word k in topic i."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$\\phi_{i_k}$", mathjax=True)),
                        html.Td("Importance of word k in topic i."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            dcc.Markdown(
                                "$\\bar{\\phi_{i}} = \\frac{\\sum_k \\phi_{i_k}}{K}$",
                                mathjax=True,
                            )
                        ),
                        html.Td("Mean importance of all words in topic i."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            dcc.Markdown(
                                "$S_{\\phi_{i}} = \\sqrt{\\frac{\\sum_k \\phi_{i_k} - \\bar{\\phi_i}}{K}}$",
                                mathjax=True,
                            )
                        ),
                        html.Td(
                            "Standard deviation of importance of all words in topic i."
                        ),
                    ]
                ),
            ],
        )
    ),
]

TIMELINE = [
    dmc.Title("Timeline", order=3),
    dmc.Text(
        """
        On the right side of the page you will find a document timeline.
        This plot displays the progression/distribution of topics in the selected document over time.
        """
    ),
    dmc.Title("How is the graph produced?", order=4),
    dmc.Text(
        """
        Values here are calculated by taking a sliding window over tokens in the document and inferring topical content
        for each window.
        Window size can be adjusted with the slider on top.
        You can highlight or hide topics by clicking at them on the legend.
        """
    ),
    dmc.Title("Topic Importances", order=3),
    dmc.Text(
        """
        The bar chart on the right displays the topical content of the document.
        Topic importances are directly taken from the document-topic-matrix.
        Only the 15 highest ranking topics are displayed on this figure.
        """
    ),
]
