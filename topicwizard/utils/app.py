"""Utilities for initializing and manipulating apps."""
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import dash


def is_notebook() -> bool:
    """Checks if code is running in a Jupyter notebook or interactive shell"""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__  # noqa
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

def get_dash():
    """Returns the appropriate Dash intarface depending on whether the code
    is running in a notebook"""
    if is_notebook():
        try:
            from jupyter_dash import JupyterDash

            Dash = JupyterDash
        except ImportError:
            warnings.warn(
                "You are running code in a Jupyter notebook, but you don't have "
                "JupyterDash installed, if you wish to use topic wizard inside "
                "Jupyter install jupyter_dash. Defaulting to server mode."
            )
            Dash = dash.Dash
    else:
        Dash = dash.Dash
    return Dash


def add_callbacks(app: dash.Dash, callbacks: List[Dict]) -> None:
    """Adds the list of callbacks to a Dash app.

    Parameters
    ----------
    app: Dash
        Dash application to add callbacks to.
    callbacks: list of dict
        Callback list to add to the app.
    """
    for callback in callbacks:
        app.callback(*callback["args"], **callback["kwargs"])(
            callback["function"]
        )

def init_callbacks() -> Tuple[List[Dict], Callable]:
    """Initialises callbacks for a module.

    Returns
    -------
    callbacks: list of dict
        List of callbacks for the module, that can be added to an app.
    decorator: function
        Function decorator that will add the function to the callback list as
        a callback.
    """
    callbacks = []
    def decorator(*args, **kwargs) -> Callable:
        def _cb(func: Callable):
            callbacks.append({"function": func, "args": args, "kwargs": kwargs})
            return func

        return _cb
    return callbacks, decorator

def get_app():
    """Initialises appropriate dash app with tailwindcss and fontawesome"""
    _dash = get_dash()
    app = _dash(
        __name__,
        title="Topic visualization",
        external_scripts=[
            {
                "src": "https://cdn.tailwindcss.com",
            },
            {
                "src": "https://kit.fontawesome.com/9640e5cd85.js",
                "crossorigin": "anonymous",
            },
        ],
    )
    return app

