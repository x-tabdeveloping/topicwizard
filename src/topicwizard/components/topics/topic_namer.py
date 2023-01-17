"""Topic namer component."""

import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint, Output, Input, State
from dash_iconify import DashIconify

topic_namer = DashBlueprint()

topic_namer.layout = dmc.TextInput(
    id="topic_namer",
    label="",
    placeholder="Rename topic...",
    size="md",
    radius="xl",
    debounce=500,
    icon=DashIconify(icon="clarity:hashtag-solid", width=15),
)

topic_namer.clientside_callback(
    """
    function (currentTopic) {
        return false;
    }
    """,
    Output("topic_namer", "disabled"),
    Input("current_topic", "data"),
)

topic_namer.clientside_callback(
    """
    function (currentTopic, topicNames) {
        if (!topicNames) {
            return '';
        }
        return topicNames[currentTopic];
    }
    """,
    Output("topic_namer", "placeholder"),
    Input("current_topic", "data"),
    State("topic_names", "data"),
)

topic_namer.clientside_callback(
    """
    function (value, currentNames, currentTopic) {
        if (value === '') {
            return currentNames;
        } else {
            const newNames = [...currentNames];
            newNames[currentTopic] = value;
            return newNames;
        }
    }
    """,
    Output("topic_names", "data"),
    Input("topic_namer", "value"),
    State("topic_names", "data"),
    State("current_topic", "data"),
)

topic_namer.clientside_callback(
    """
    function (currentTopic) {
        return '';
    }
    """,
    Output("topic_namer", "value"),
    Input("current_topic", "data"),
)
