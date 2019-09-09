import numpy as np


def get_colors(color_st, color_fi, num):
    """Gets gradient colors.

    Parameters
    ----------
    color_st : tuple of 3 float from 0 to 1
        Initial color.
    color_fi : tuple of 3 float from 0 to 1
        Final color.
    num : int
        Number of colors.

    Returns
    -------
    list of tuples of 3 float from 0 to 1
        Sequence of colors from `color_st` to `color_fi`.

    """
    colors = [(cx, cy, cz) for cx, cy, cz in
              zip(np.linspace(color_st[0], color_fi[0], num),
                  np.linspace(color_st[1], color_fi[1], num),
                  np.linspace(color_st[2], color_fi[2], num))]

    return colors


def get_ticks(min_values, max_values, max_num):
    """ Gets beauty ticks for graphics.

    Parameters
    ----------
    min_values : int or float
        Minimal value.
    max_values : int or float
        Maximum values.
    max_num:
        Maximal number of ticks.

    Returns
    -------
    np.ndarray, 1d
        Ticks.

    """
    step = int(np.ceil((max_values - min_values) / max_num))
    if step <= 1:
        step = 1
    else:
        step = int(np.ceil(step / 5) * 5)
    st = int(np.floor(min_values + (max_values - min_values
                                    - step * np.ceil((max_values - min_values) / step)) / 2.))
    fi = st + int(step * np.ceil((max_values - min_values) / step))
    if step == 1:
        st = int(np.floor(st))
        fi = int(np.ceil(fi))
    else:
        fi = fi - (st - int(np.floor(st / 5) * 5))
        st = int(np.floor(st / 5) * 5)
        if fi < max_values:
            if step // 5 % 2:
                fi += (step // 5 // 2 + 1) * 5
                st -= (step // 5 // 2) * 5
            else:
                fi += (step // 5 // 2) * 5
                st -= (step // 5 // 2) * 5
    if st + step <= min_values:
        st = st + step
    ticks = np.arange(st, fi + 1, step=step).astype(int)

    return ticks
