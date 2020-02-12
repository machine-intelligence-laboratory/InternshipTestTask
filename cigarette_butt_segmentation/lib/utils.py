import cv2
import numpy as np


def get_mask(img_id, annotations):
    """Returns mask.

    Parameters
    ----------
    img_id : int
        Image id.
    annotations : dict
        Ground truth.

    Returns
    -------
    np.ndarray, 2d
        Mask that contains only 2 unique values: 0 - denotes background, 255 - denotes object.
    
    """
    img_info = annotations["images"][img_id]
    assert img_info["id"] == img_id
    w, h = img_info["width"], img_info["height"]
    mask = np.zeros((h, w)).astype(np.uint8)
    gt = annotations["annotations"][img_id]
    assert gt["id"] == img_id
    polygon = np.array(gt["segmentation"][0]).reshape((-1, 2))
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=255)
    
    return mask


def encode_rle(mask):
    """Returns encoded mask (run length) as a string.

    Parameters
    ----------
    mask : np.ndarray, 2d
        Mask that consists of 2 unique values: 0 - denotes background, 1 - denotes object.

    Returns
    -------
    str
        Encoded mask.

    Notes
    -----
    Mask should contains only 2 unique values, one of them must be 0, another value, that denotes
    object, could be different from 1 (for example 255).

    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def decode_rle(rle_mask, shape=(512, 512)):
    """Decodes mask from rle string.

    Parameters
    ----------
    rle_mask : str
        Run length as string formatted.
    shape : tuple of 2 int, optional (default=(320, 240))
        Shape of the decoded image.

    Returns
    -------
    np.ndarray, 2d
        Mask that contains only 2 unique values: 0 - denotes background, 255 - denotes object.
    
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 255

    return img.reshape(shape)
