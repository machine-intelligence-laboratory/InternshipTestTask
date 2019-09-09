import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

def extend_arr(t, arr, t_inter=None, t_min=None, t_max=None):
    """Extends arr values according timestamps `t`.

    Function interpolate values between neighbouring timestamps. According to this procedure
    the function obtains values in each timestamp between min and max timestamp.

    Notes
    -----
    The function interpolates the values with step 1 between neighbouring timestamps.

    Parameters
    ----------
    t : array_like of int, 1d
        One dimensional array_like with timestamps.
    arr : array_like, 1d
        One dimensional array_like object.
    t_inter : array_like, 1d or None, optional (default=None)
        If it is not None, then the function interpolates values on `t_inter`.
    t_min : int, optional (default=None)
        Minimal timestamp. If None then calculates min of `t`. Used when `t_inter` is None.
    t_max : int, optional (default=None)
        Maximal timestamp. If None then calculates max of `t`. Used when `t_inter` is None.

    Returns
    -------
    np.ndarray, 1d
        Extended timestamps.
    np.ndarray, 1d
        Extended values.

    """
    if t_inter is None:
        if t_min is None:
            t_min = t.min()
        if t_max is None:
            t_max = t.max()
        extended_t = np.arange(t_min, t_max + 1)
    else:
        extended_t = t_inter
    extended_values = np.interp(extended_t, t, arr)

    return extended_t, extended_values


def extend_data(data, column_by="tmsp", extended_columns=("x", "y", "z"),
                t_inter=None, t_min=None, t_max=None):
    """Extends x, y, z axes values (`extended_columns`) according timestamps `t_column`.

    Function interpolate values between neighbouring timestamps for each column
    in `extended_columns` separately. According to this procedure the function obtains values
    in each timestamp between min and max timestamp.

    Notes
    -----
    The function interpolates the values with step 1 between neighbouring timestamps.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with columns `extended_columns` and `column_by`.
    column_by : str
        Column of the `data` containing timestamps.
    extended_columns : tuple of str
        Columns of the `data` to be extended.
    t_inter : array_like, 1d or None, optional (default=None)
        If it is not None, then the function interpolates values on `t_inter`.
    t_min : int, optional (default=None)
        Minimal timestamp. If None then calculates min of `t`. Used when `t_inter` is None.
    t_max : int, optional (default=None)
        Maximal timestamp. If None then calculates max of `t`. Used when `t_inter` is None.

    Returns
    -------
    pd.DataFrame
        Dataframe with extended values in columns `extended_columns` and `column_by`.

    """
    data = data.copy()
    extended_data = dict()
    t = data.loc[:, column_by].values.copy()
    for column in extended_columns:
        extended_t, extended_values = extend_arr(t, data.loc[:, column].values,
                                                 t_inter, t_min, t_max)
        if column_by not in extended_data:
            extended_data[column_by] = extended_t
        extended_data[column] = extended_values
    extended_data = pd.DataFrame(extended_data)

    return extended_data
