import dash_mantine_components as dmc
from dash_extensions.enrich import dcc, html

GROUP_MAP = [
    dmc.Title("Group Map", order=3),
    dmc.Text(
        "On the left side of this page you will find the group map. It will help you examine label importances and intertopic distances.",
    ),
    dmc.Text("To select a label to investigate, click on it on the map."),
    dmc.Title("How are positions calculated?", order=4),
    dmc.Text(
        """
        Distances are based on a group-topic-importance matrix.
        embeddings from this matrix are projected into 2-dimensional space using UMAP.
        """
    ),
    dmc.Title("How are sizes calculated?", order=4),
    dmc.Text(
        "Sizes on the map are proportional to the amount of documents belonging to a given group."
    ),
    dmc.Title("How are groups colored?", order=4),
    dmc.Text(
        "Colors are determined by the most important topic in a given group. Each topic has a different color."
    ),
    dmc.Table(
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$c_j = argmax_i g_{j_i}$", mathjax=True)),
                        html.Td("Most important topic for group j"),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(
                            dcc.Markdown(
                                "$g_{j_i} = \\sum_{d \\in G} d_{i}$", mathjax=True
                            )
                        ),
                        html.Td("Importance of topic i for group j."),
                    ]
                ),
                html.Tr(
                    [
                        html.Td(dcc.Markdown("$d_{i}$", mathjax=True)),
                        html.Td("Importance of topic i for a given document."),
                    ]
                ),
            ],
        )
    ),
    dmc.Title("Bar Chart", order=3),
    dmc.Text(
        "Importances of topics calculated as outlined above are displayed on a bar chart in the middle.",
    ),
    dmc.Title("Important Words", order=3),
    dmc.Text(
        """
        On the right you will find a wordcloud displaying the most frequent words in a group.
        Most frequent words are obtained by summing up the document-term-matrix in each group.
        """
    ),
]
