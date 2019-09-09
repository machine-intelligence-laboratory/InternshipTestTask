import math
import numpy as np
import pandas as pd

from lib.preprocess import extend_data

def synchronize_trajectories(trajectory_true, trajectory_pred,
                             t_column="tmsp", coordinate_columns=("x", "y")):
    """Synchronizes trajectories using timestamps.

    Parameters
    ----------
    trajectory_true : pd.DataFrame
        Dataframe with columns `t_column` + `coordinate_columns` contains ground truth coordinates
        of trajectory.
    trajectory_pred : pd.DataFrame
        Dataframe with columns `t_column` + `coordinate_columns` contains predicted coordinates
        of trajectory.
    t_column : str, optional (default="tmsp")
        Column of both dataframes that denotes timestamps.
    coordinate_columns : tuple of 2 str, optional (default=("x", "y"))
        Two columns of both dataframes that denote X and Y axes coordinates.

    Returns
    -------
    pd.DataFrame
        Ground truth without changes.
    pd.DataFrame
        Predicted trajectory coordinates at the timestamps from ground truth.

    """
    t_true = trajectory_true.loc[:, t_column].values
    trajectory_pred = extend_data(trajectory_pred, t_inter=t_true,
                                  column_by=t_column, extended_columns=coordinate_columns)

    return trajectory_true, trajectory_pred


def straighten_trajectories(trajectory_true, trajectory_pred, coordinate_columns=("x", "y")):
    """Straightens true trajectory and calculates distances between trajectories.

    Parameters
    ----------
    trajectory_true : pd.DataFrame
        Dataframe with columns `t_column` + `coordinate_columns` contains ground truth coordinates
        of trajectory.
    trajectory_pred : pd.DataFrame
        Dataframe with columns `t_column` + `coordinate_columns` contains predicted coordinates
        of trajectory.
    coordinate_columns : tuple of 2 str, optional (default=("x", "y"))
        Two columns of both dataframes that denote X and Y axes coordinates.

    Returns
    -------
    np.ndarray, 1d
        Coordinates on X axis.
    np.ndarray, 1d
        Coordinates on Y axis. Distances between trajectories.

    """
    x_true = trajectory_true.loc[:, coordinate_columns[0]].values
    y_true = trajectory_true.loc[:, coordinate_columns[1]].values
    x_pred = trajectory_pred.loc[:, coordinate_columns[0]].values
    y_pred = trajectory_pred.loc[:, coordinate_columns[1]].values
    if x_true.shape[0] != x_pred.shape[0]:
        raise ValueError("Shape of true and pred array should be the same.")
    y = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)
    x = np.cumsum(np.array([0] + [np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) for x1, x2, y1, y2
                                  in zip(x_true[:-1], x_true[1:],
                                         y_true[:-1], y_true[1:])]))

    return x, y
