"""Plotting utilities/utility plots"""

from pathlib import Path
from urllib.request import urlretrieve

import plotly.express as px


def text_plot(text: str):
    """Returns empty scatter plot with text added, this can be great for error messages."""
    return px.scatter().add_annotation(text=text, showarrow=False, font=dict(size=20))


def get_default_font_path() -> Path:
    """Returns path for Open Sans font file.
    Downloads the file if needed.
    """
    fonts_dir = Path.home().joinpath(".topicwizard", "fonts")
    fonts_dir.mkdir(exist_ok=True, parents=True)
    path = fonts_dir.joinpath("OpenSans-Bold.otf")
    try:
        if not path.is_file():
            urlretrieve(
                "https://github.com/googlefonts/opensans/raw/refs/heads/main/fonts/ttf/OpenSans-Bold.ttf",
                path,
            )
    except Exception:
        return None
    return path
