# File content is auto-generated. Do not modify.
# pylint: skip-file
from ._internal import NDArrayBase
from ..base import _Null

def adjust_lighting(data=None, alpha=_Null, out=None, name=None, **kwargs):
    r"""Adjust the lighting level of the input. Follow the AlexNet style.

    Defined in src/operator/image/image_random.cc:L242

    Parameters
    ----------
    data : NDArray
        The input.
    alpha : tuple of <float>, required
        The lighting alphas for the R, G, B channels.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def crop(data=None, x=_Null, y=_Null, width=_Null, height=_Null, out=None, name=None, **kwargs):
    r"""Crop an image NDArray of shape (H x W x C) or (N x H x W x C) 
    to the given size.
    Example:
        .. code-block:: python
            image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
            mx.nd.image.crop(image, 1, 1, 2, 2)
                [[[144  34   4]
                  [ 82 157  38]]

                 [[156 111 230]
                  [177  25  15]]]
                <NDArray 2x2x3 @cpu(0)>
            image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
            mx.nd.image.crop(image, 1, 1, 2, 2)            
                [[[[ 35 198  50]
                   [242  94 168]]

                  [[223 119 129]
                   [249  14 154]]]


                  [[[137 215 106]
                    [ 79 174 133]]

                   [[116 142 109]
                    [ 35 239  50]]]]
                <NDArray 2x2x2x3 @cpu(0)>


    Defined in src/operator/image/crop.cc:L65

    Parameters
    ----------
    data : NDArray
        The input.
    x : int, required
        Left boundary of the cropping area.
    y : int, required
        Top boundary of the cropping area.
    width : int, required
        Width of the cropping area.
    height : int, required
        Height of the cropping area.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def flip_left_right(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L192

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def flip_top_bottom(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L200

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def normalize(data=None, mean=_Null, std=_Null, out=None, name=None, **kwargs):
    r"""Normalize an tensor of shape (C x H x W) or (N x C x H x W) with mean and
        standard deviation.

        Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,
        this transform normalizes each channel of the input tensor with:

    .. math::

            output[i] = (input[i] - m\ :sub:`i`\ ) / s\ :sub:`i`

        If mean or std is scalar, the same value will be applied to all channels.

        Default value for mean is 0.0 and stand deviation is 1.0.

    Example:

        .. code-block:: python
            image = mx.nd.random.uniform(0, 1, (3, 4, 2))
            normalize(image, mean=(0, 1, 2), std=(3, 2, 1))
                [[[ 0.18293785  0.19761486]
                  [ 0.23839645  0.28142193]
                  [ 0.20092112  0.28598186]
                  [ 0.18162774  0.28241724]]
                 [[-0.2881726  -0.18821815]
                  [-0.17705294 -0.30780914]
                  [-0.2812064  -0.3512327 ]
                  [-0.05411351 -0.4716435 ]]
                 [[-1.0363373  -1.7273437 ]
                  [-1.6165586  -1.5223348 ]
                  [-1.208275   -1.1878313 ]
                  [-1.4711051  -1.5200229 ]]]
                <NDArray 3x4x2 @cpu(0)>

            image = mx.nd.random.uniform(0, 1, (2, 3, 4, 2))
            normalize(image, mean=(0, 1, 2), std=(3, 2, 1))
                [[[[ 0.18934818  0.13092826]
                   [ 0.3085322   0.27869293]
                   [ 0.02367868  0.11246539]
                   [ 0.0290431   0.2160573 ]]
                  [[-0.4898908  -0.31587923]
                   [-0.08369008 -0.02142242]
                   [-0.11092162 -0.42982462]
                   [-0.06499392 -0.06495637]]
                  [[-1.0213816  -1.526392  ]
                   [-1.2008414  -1.1990893 ]
                   [-1.5385206  -1.4795225 ]
                   [-1.2194707  -1.3211205 ]]]
                 [[[ 0.03942481  0.24021089]
                   [ 0.21330701  0.1940066 ]
                   [ 0.04778443  0.17912441]
                   [ 0.31488964  0.25287187]]
                  [[-0.23907584 -0.4470462 ]
                   [-0.29266903 -0.2631998 ]
                   [-0.3677222  -0.40683383]
                   [-0.11288315 -0.13154092]]
                  [[-1.5438497  -1.7834496 ]
                   [-1.431566   -1.8647819 ]
                   [-1.9812102  -1.675859  ]
                   [-1.3823645  -1.8503251 ]]]]
                <NDArray 2x3x4x2 @cpu(0)>


    Defined in src/operator/image/image_random.cc:L165

    Parameters
    ----------
    data : NDArray
        Input ndarray
    mean : tuple of <float>, optional, default=[0,0,0,0]
        Sequence of means for each channel. Default value is 0.
    std : tuple of <float>, optional, default=[1,1,1,1]
        Sequence of standard deviations for each channel. Default value is 1.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_brightness(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L208

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_color_jitter(data=None, brightness=_Null, contrast=_Null, saturation=_Null, hue=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L235

    Parameters
    ----------
    data : NDArray
        The input.
    brightness : float, required
        How much to jitter brightness.
    contrast : float, required
        How much to jitter contrast.
    saturation : float, required
        How much to jitter saturation.
    hue : float, required
        How much to jitter hue.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_contrast(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L214

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_flip_left_right(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L196

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_flip_top_bottom(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L204

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_hue(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L228

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_lighting(data=None, alpha_std=_Null, out=None, name=None, **kwargs):
    r"""Randomly add PCA noise. Follow the AlexNet style.

    Defined in src/operator/image/image_random.cc:L249

    Parameters
    ----------
    data : NDArray
        The input.
    alpha_std : float, optional, default=0.0500000007
        Level of the lighting noise.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_saturation(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L221

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def resize(data=None, size=_Null, keep_ratio=_Null, interp=_Null, out=None, name=None, **kwargs):
    r"""Resize an image NDArray of shape (H x W x C) or (N x H x W x C) 
    to the given size
    Example:
        .. code-block:: python
            image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
            mx.nd.image.resize(image, (3, 3))
                [[[124 111 197]
                  [158  80 155]
                  [193  50 112]]

                 [[110 100 113]
                  [134 165 148]
                  [157 231 182]]

                 [[202 176 134]
                  [174 191 149]
                  [147 207 164]]]
                <NDArray 3x3x3 @cpu(0)>
            image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
            mx.nd.image.resize(image, (2, 2))            
                [[[[ 59 133  80]
                   [187 114 153]]

                  [[ 38 142  39]
                   [207 131 124]]]


                  [[[117 125 136]
                   [191 166 150]]

                  [[129  63 113]
                   [182 109  48]]]]
                <NDArray 2x2x2x3 @cpu(0)>


    Defined in src/operator/image/resize.cc:L70

    Parameters
    ----------
    data : NDArray
        The input.
    size : Shape(tuple), optional, default=[]
        Size of new image. Could be (width, height) or (size)
    keep_ratio : boolean, optional, default=0
        Whether to resize the short edge or both edges to `size`, if size is give as an integer.
    interp : int, optional, default='1'
        Interpolation method for resizing. By default uses bilinear interpolationOptions are INTER_NEAREST - a nearest-neighbor interpolationINTER_LINEAR - a bilinear interpolationINTER_AREA - resampling using pixel area relationINTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhoodINTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhoodNote that the GPU version only support bilinear interpolation(1) and the result on cpu would be slightly different from gpu.It uses opencv resize function which tend to align center on cpuwhile using contrib.bilinearResize2D which aligns corner on gpu

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def to_tensor(data=None, out=None, name=None, **kwargs):
    r"""Converts an image NDArray of shape (H x W x C) or (N x H x W x C) 
    with values in the range [0, 255] to a tensor NDArray of shape (C x H x W) or (N x C x H x W)
    with values in the range [0, 1]

    Example:
        .. code-block:: python
            image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
            to_tensor(image)
                [[[ 0.85490197  0.72156864]
                  [ 0.09019608  0.74117649]
                  [ 0.61960787  0.92941177]
                  [ 0.96470588  0.1882353 ]]
                 [[ 0.6156863   0.73725492]
                  [ 0.46666667  0.98039216]
                  [ 0.44705883  0.45490196]
                  [ 0.01960784  0.8509804 ]]
                 [[ 0.39607844  0.03137255]
                  [ 0.72156864  0.52941179]
                  [ 0.16470589  0.7647059 ]
                  [ 0.05490196  0.70588237]]]
                 <NDArray 3x4x2 @cpu(0)>

            image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
            to_tensor(image)
                [[[[0.11764706 0.5803922 ]
                   [0.9411765  0.10588235]
                   [0.2627451  0.73333335]
                   [0.5647059  0.32156864]]
                  [[0.7176471  0.14117648]
                   [0.75686276 0.4117647 ]
                   [0.18431373 0.45490196]
                   [0.13333334 0.6156863 ]]
                  [[0.6392157  0.5372549 ]
                   [0.52156866 0.47058824]
                   [0.77254903 0.21568628]
                   [0.01568628 0.14901961]]]
                 [[[0.6117647  0.38431373]
                   [0.6784314  0.6117647 ]
                   [0.69411767 0.96862745]
                   [0.67058825 0.35686275]]
                  [[0.21960784 0.9411765 ]
                   [0.44705883 0.43529412]
                   [0.09803922 0.6666667 ]
                   [0.16862746 0.1254902 ]]
                  [[0.6156863  0.9019608 ]
                   [0.35686275 0.9019608 ]
                   [0.05882353 0.6509804 ]
                   [0.20784314 0.7490196 ]]]]
                <NDArray 2x3x4x2 @cpu(0)>


    Defined in src/operator/image/image_random.cc:L91

    Parameters
    ----------
    data : NDArray
        Input ndarray

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

__all__ = ['adjust_lighting', 'crop', 'flip_left_right', 'flip_top_bottom', 'normalize', 'random_brightness', 'random_color_jitter', 'random_contrast', 'random_flip_left_right', 'random_flip_top_bottom', 'random_hue', 'random_lighting', 'random_saturation', 'resize', 'to_tensor']