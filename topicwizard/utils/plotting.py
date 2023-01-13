"""Module for graphing utilites and hacks"""
from typing import Tuple

import numpy as np


def _get_edge_pos(edges: np.ndarray, x_y: np.ndarray) -> np.ndarray:
    """
    Transforms edges and either the x or the y positions of nodes to
    the x or y positions for the lines in the plotly figure.
    """
    # WARNING: Nasty numpy tricks
    # Getting positions of the points for each end of the edges
    end_positions = x_y[edges]
    # Creating an array with +1 dimension along the first axis, for inserting
    # nans
    x_y_edges = np.zeros((end_positions.shape[0], end_positions.shape[1] + 1))
    x_y_edges[:, :-1] = end_positions
    # In order for the line not to be connected, we have to insert a nan
    # after each pair of points that have to be connected.
    x_y_edges[:, -1] = np.nan
    # The result has to be of rank 1, so we flatten the array.
    return x_y_edges.flatten()


def get_edge_positions(
    edges: np.ndarray, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms edges and node positions to edge positions that plotly expects.

    Parameters
    ----------
    edges: ndarray of shape (n_edges, 2)
        Edges represented as pairs of node indices.
    x: ndarray of shape (n_nodes, )
        X positions of the nodes.
    y: ndarray of shape (n_nodes, )
        Y positions of the nodes.

    Returns
    -------
    edge_x: ndarray of shape (n_edges, )
        X positions of the edges.
    edge_y: ndarray of shape (n_edges, )
        Y positions of the edges.
    """
    edge_x = _get_edge_pos(edges, x)
    edge_y = _get_edge_pos(edges, y)
    return edge_x, edge_y
