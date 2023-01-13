from dash_extensions.enrich import (
    DashBlueprint,
    dcc,
)

wordcloud = DashBlueprint()

wordcloud.layout = dcc.Graph(
    id="wordcloud",
    responsive=True,
    className="flex-1",
    config=dict(scrollZoom=True),
)
