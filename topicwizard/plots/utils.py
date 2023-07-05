"""Plotting utilities/utility plots"""
import plotly.express as px


def text_plot(text: str):
    """Returns empty scatter plot with text added, this can be great for error messages."""
    return px.scatter().add_annotation(text=text, showarrow=False, font=dict(size=20))
