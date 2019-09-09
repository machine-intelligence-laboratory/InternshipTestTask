import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from lib.postprocess import synchronize_trajectories, straighten_trajectories


def rmse(true_data, pred_data, t_column="tmsp", coordinate_columns=("x", "y")):
    """Root mean square error regression loss.

    Parameters
    ----------
    true_data : pd.DataFrame or array_like, 1d
        Dataframe with columns `t_column` + `coordinate_columns` contains ground truth coordinates
        of trajectory or array-like object.
    pred_data : pd.DataFrame or array_like, 1d
        Dataframe with columns `t_column` + `coordinate_columns` contains predicted coordinates
        of trajectory or array-like object.
    t_column : str, optional (default="tmsp")
        It is needed if dataframes are passed. Column of both dataframes that denotes timestamps.
    coordinate_columns : tuple of 2 str, optional (default=("x", "y"))
        It is needed if dataframes are passed. Two columns of both dataframes that denote X and
        Y axes coordinates.

    Returns
    -------
    float
        A non-negative floating point value (the best value is 0.0).

    """
    if type(true_data) != type(pred_data):
        raise TypeError("True and pred data should have the same type.")
    if isinstance(true_data, pd.DataFrame):
        true_data, pred_data = synchronize_trajectories(
            true_data, pred_data, t_column=t_column, coordinate_columns=coordinate_columns)
        x_true = true_data.loc[:, coordinate_columns[0]].values
        y_true = true_data.loc[:, coordinate_columns[1]].values
        x_pred = pred_data.loc[:, coordinate_columns[0]].values
        y_pred = pred_data.loc[:, coordinate_columns[1]].values
        pred_data = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)
        true_data = np.zeros(y_pred.shape[0])
    error = np.sqrt(mean_squared_error(true_data, pred_data))

    return error


def mie(trajectory_true, trajectory_pred, t_column="tmsp", coordinate_columns=("x", "y")):
    """Mean integral distance.

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
    float
        Mean integral error (the best value is 0.0).

    """
    trajectory_true, trajectory_pred = synchronize_trajectories(
        trajectory_true, trajectory_pred,
        t_column=t_column, coordinate_columns=coordinate_columns)
    x, y = straighten_trajectories(trajectory_true, trajectory_pred,
                                   coordinate_columns=coordinate_columns)
    error = ((y[1:] + y[:-1]) / 2 * (x[1:] - x[:-1])).sum() / x[-1]

    return error


def get_scores(trajectory_true, trajectory_pred, t_column="tmsp", coordinate_columns=("x", "y")):
    """Returns rmse, mean integral distance, and final gap.

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
    dict[str:float]
        Dict with rmse, mie, and gap keys which are represented:
        * Root mean square error regression loss;
        * Mean integral distance;
        * Distance between first and last points.

    """
    scores = {
        "rmse": rmse(trajectory_true, trajectory_pred, t_column, coordinate_columns),
        "mie": mie(trajectory_true, trajectory_pred, t_column, coordinate_columns),
    }

    return scores
