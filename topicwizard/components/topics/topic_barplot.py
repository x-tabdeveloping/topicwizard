from dash_extensions.enrich import DashBlueprint, dcc

topic_barplot = DashBlueprint()

topic_barplot.layout = dcc.Graph(
    id="topic_barplot",
    responsive=True,
    animate=False,
    className="flex-1",
)
