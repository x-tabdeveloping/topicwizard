"""Topic switcher components"""
from dash import html

button_class = """
    text-xl transition-all ease-in 
    justify-center content-center items-center
    text-gray-500 hover:text-sky-600
    flex flex-1
"""

mini_switcher = html.Div(
    className="""
        fixed flex flex-none flex-row justify-center content-middle
        left-0.5 bottom-5 h-16 w-32 bg-white shadow rounded-full
        rounded-full ml-5
    """,
    children=[
        html.Button(
            "<-",
            id="prev_topic",
            title="Switch to previous topic",
            className=button_class,
        ),
        html.Button(
            "->",
            id="next_topic",
            title="Switch to next topic",
            className=button_class,
        ),
    ],
)
