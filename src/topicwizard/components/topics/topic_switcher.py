"""Topic switcher component."""
import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint, Input, Output, State
from dash_iconify import DashIconify

topic_switcher = DashBlueprint()

topic_switcher.layout = dmc.Group(
    [
        dmc.ActionIcon(
            DashIconify(icon="material-symbols:chevron-left", width=30),
            id="prev_topic",
            size="xl",
            color="orange",
            variant="subtle",
            radius="xl",
            n_clicks=0,
            disabled=True,
        ),
        dmc.ActionIcon(
            DashIconify(icon="material-symbols:chevron-right", width=30),
            id="next_topic",
            size="xl",
            color="orange",
            variant="subtle",
            radius="xl",
            n_clicks=0,
            disabled=False,
        ),
    ],
    position="left",
    grow=1,
)

# Disable next button if already at last topic
topic_switcher.clientside_callback(
    """
    function(currentTopic, topicNames) {
        return topicNames.length - 1 <= currentTopic
    }
    """,
    Output("next_topic", "disabled"),
    Input("current_topic", "data"),
    State("topic_names", "data"),
)

# Disable previous button if already at first topic
topic_switcher.clientside_callback(
    """
    function(currentTopic) {
        return currentTopic <= 0
    }
    """,
    Output("prev_topic", "disabled"),
    Input("current_topic", "data"),
)
