from typing import Optional, Tuple

import dash
from dash import html
from dash.dependencies import Input, Output

from topicwizard.utils.app import init_callbacks

settings_visible = "flex-1 flex-col flex items-stretch justify-evenly"
settings_hidden = "hidden"

accordion_callbacks, cb = init_callbacks()


def Accordion(
    name: str,
    children,
    index: Optional[str] = None,
) -> html.Div:
    """Accordion component for internal use in the application

    Parameters
    ----------
    name: str
        Title of the accordion.
    children: dash element or list of dash elements
        Children of the accordion element.
    index: str or None, default None
        Index to be used for the created accordion's callbacks.
        Defaults to the name of the accordion, duplicate names could pose a
        problem though.

    Returns
    -------
    Div
        Dash div element.
    """
    if index is None:
        index = name
    return html.Div(
        children=[
            html.Div(
                [
                    html.H3(
                        name,
                        className="text-xl",
                    ),
                    html.Span(className="flex-1"),
                    html.Button(
                        html.I(
                            className="fa-solid fa-chevron-up",
                        ),
                        id=dict(type="_accordion_collapse", index=index),
                        n_clicks=0,
                    ),
                ],
                className="""
                flex flex-row justify-center content-center
                px-5 py-1
                """,
            ),
            html.Span(
                className="""
                block bg-gray-500 bg-opacity-10 h-0.5 self-center m-2
                """
            ),
            html.Div(
                id=dict(type="_accordion_body", index=index),
                className=settings_visible,
                children=children,
            ),
        ],
        className="""
            justify-center content-center
            bg-white p-3 rounded-2xl shadow
            transition-all ease-in 
        """,
    )


def AccordionItem(name: str, *children):
    """Accordion item with a title and a child component"""
    return html.Div(
        className="""
            flex flex-row flex-1
            justify-between justify-items-stretch
            content-center items-center
            px-4 my-1.5
            """,
        children=[
            html.P(name),
            *children,
        ],
    )


@cb(
    Output(
        dict(type="_accordion_body", index=dash.MATCH),
        "className",
    ),
    Output(
        dict(type="_accordion_collapse", index=dash.MATCH),
        "className",
    ),
    Input(dict(type="_accordion_collapse", index=dash.MATCH), "n_clicks"),
    prevent_initial_call=True,
)
def expand_hide_accordion(n_clicks: int) -> Tuple[str, str]:
    is_on = not (n_clicks % 2)
    if is_on:
        return settings_visible, "transition-all ease-in rotate-0"
    else:
        return "hidden", "transition-all ease-in rotate-180"
