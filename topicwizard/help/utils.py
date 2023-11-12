import dash_mantine_components as dmc
from dash_extensions.enrich import html
from dash_iconify import DashIconify


def make_helper(content, width="500px"):
    return dmc.HoverCard(
        width=width,
        withArrow=False,
        style={"p": "5px", "m": "3px"},
        shadow="lg",
        children=[
            dmc.HoverCardTarget(
                html.Div(
                    DashIconify(icon="uil:question", color="black", width=45),
                    className="rounded-full border-black border-4",
                ),
            ),
            dmc.HoverCardDropdown(content),
        ],
    )
