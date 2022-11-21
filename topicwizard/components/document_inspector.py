from dash import dcc, html

from topicwizard.components.accordion import Accordion

document_inspector = html.Div(
    className="""basis-1/3 flex-1 flex-col bg-white shadow
    overflow-y-scroll overflow-x-hidden p-5 space-y-5
    """,
    children=[
        dcc.Dropdown(
            id="document_selector",
            options={},
            value=None,
        ),
        Accordion(
            "Topics",
            index="inspector_topics",
            children=[
                dcc.Graph(id="document_topics_graph", animate=False),
            ],
        ),
        Accordion(
            "Content",
            index="inspector_content",
            children=[
                html.Div(
                    id="document_content",
                    children="This is the textual content of the document",
                    className="""
                text-justify h-1/3
                """,
                ),
            ],
        ),
    ],
)
