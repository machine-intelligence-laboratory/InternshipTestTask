import numpy as np
import matplotlib.pyplot as plt

from lib.metrics import get_scores, rmse
from lib.utils import get_colors, get_ticks
from lib.postprocess import synchronize_trajectories, straighten_trajectories

def show_trajectories(trajectory_true, trajectory_pred,
                      t_column="tmsp", coordinate_columns=("x", "y"),
                      coef=0.1, figsize=(16, 4.5),
                      pred_color_st=(0, 0, 1), pred_color_fi=(0, 0, 0.3),
                      true_color_st=(0, 1, 0), true_color_fi=(0, 0.3, 0)):
    """Shows true and predicted trajectories with metrics.

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
    coef : float from 0 to 1, optional(default=0.1)

    figsize : tuple of 2 int, optional (default=(8, 8))
        Size of the figure.
    pred_color_st : tuple of 3 float from 0 to 1, optional (default=(0, 0, 1))
        Initial color for predicted color.
    pred_color_fi : tuple of 3 float from 0 to 1, optional (default=(0, 0, 0.3))
        Final color for predicted color.
    true_color_st : tuple of 3 float from 0 to 1, optional (default=(0, 1, 0))
        Initial color for ground truth color.
    true_color_fi : tuple of 3 float from 0 to 1, optional (default=(0, 0.3, 1))
        Final color for ground truth color.

    Returns
    -------
    figure
        Figure - matplotlib.pyplot object.
    dict[str:float]
        Dict with rmse, mie, and gap keys which are represented:
        * Root mean square error regression loss;
        * Mean integral distance;
        * Distance between first and last points.

    """
    trajectory_true, trajectory_pred = synchronize_trajectories(
        trajectory_true, trajectory_pred, t_column=t_column, coordinate_columns=coordinate_columns)
    x_true = trajectory_true.loc[:, coordinate_columns[0]].values
    y_true = trajectory_true.loc[:, coordinate_columns[1]].values
    x_pred = trajectory_pred.loc[:, coordinate_columns[0]].values
    y_pred = trajectory_pred.loc[:, coordinate_columns[1]].values

    steps_num = min(50, len(x_pred) // 2)
    colors_pred = get_colors(pred_color_st, pred_color_fi, num=steps_num)
    colors_true = get_colors(true_color_st, true_color_fi, num=steps_num)

    legend_size = figsize[0]
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(1, 3, wspace=0.39)
    ax1, ax2 = plt.subplot(grid[0, 0]), plt.subplot(grid[0, 1:])
    # AX1 plotting
    seps = np.linspace(0, len(x_true), steps_num).astype(int)
    for ind, (sep1, sep2) in enumerate(zip(seps[:-1], seps[1:])):
        ax1.plot(x_true[sep1:sep2 + 1], y_true[sep1:sep2 + 1], color=colors_true[ind],
                 linestyle="--", linewidth=3)
    seps = np.linspace(0, len(x_pred), steps_num).astype(int)
    for ind, (sep1, sep2) in enumerate(zip(seps[:-1], seps[1:])):
        ax1.plot(x_pred[sep1:sep2 + 1], y_pred[sep1:sep2 + 1], color=colors_pred[ind], linewidth=3)

    ax1.tick_params(axis='both', which='major', labelsize=legend_size - 3, direction="in")
    ax1.grid()
    ax1.set_xlabel("X axis, meters", fontsize=legend_size)
    ax1.set_ylabel("Y axis, meters", fontsize=legend_size)

    xticks = get_ticks(min_values=min(min(x_true), min(x_pred)),
                       max_values=max(max(x_true), max(x_pred)),
                       max_num=5)
    yticks = get_ticks(min_values=min(min(y_true), min(y_pred)),
                       max_values=max(max(y_true), max(y_pred)),
                       max_num=5)
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.yaxis.set_label_coords(-0.15, 0.5)

    x, y = straighten_trajectories(trajectory_true, trajectory_pred,
                                   coordinate_columns=coordinate_columns)
    scores = get_scores(trajectory_true, trajectory_pred,
                                   coordinate_columns=coordinate_columns)
    # AX2 plotting
    seps = np.linspace(0, len(y), steps_num).astype(int)
    for ind, (sep1, sep2) in enumerate(zip(seps[:-1], seps[1:])):
        ax2.plot(x[sep1:sep2 + 1], y[sep1:sep2 + 1], color=colors_pred[ind], linewidth=2)

    ax2.plot([0, min(y.max() / coef, x.max())], [0, coef * min(y.max() / coef, x.max())],
             linestyle="--", c="g", linewidth=3)
    ax2.fill_between(x, 0, y.max(), where=y < np.linspace(0, coef * x.max(), len(y)),
                     facecolor='green', alpha=0.15)

    ax2.tick_params(axis='both', which='major', direction="in", labelsize=legend_size - 3)
    ax2.grid()
    ax2.set_xlabel("Trajectory length, meters", fontsize=legend_size)
    ax2.set_ylabel("Distance between\n trajectories, meters", fontsize=legend_size)
    ax2.yaxis.set_label_coords(-0.05, 0.5)

    yticks = get_ticks(min_values=0, max_values=max(y), max_num=10)
    xticks = get_ticks(min_values=0, max_values=max(x), max_num=10)
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.text(0.05, 0.9, "$\mathcal{L}_{tr}\,=\,$" + f"{scores['rmse']:.1f}",
             horizontalalignment='left', verticalalignment='top', fontsize=legend_size - 1,
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
    ax2.text(0.05, 0.78, "$\mathcal{D}_{tr}\,=\,$" + f"{scores['mie']:.1f}",
             horizontalalignment='left', verticalalignment='top', fontsize=legend_size - 1,
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
    plt.show()

    return fig, scores
