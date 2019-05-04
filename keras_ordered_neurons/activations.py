from keras.activations import softmax
import keras.backend as K


__all__ = ['cumax']


def cumax(x, axis=-1):
    """Cumulative sum of softmax activation.

    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the operation is applied.

    # Returns
        Tensor, output of softmax transformation.

    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 1:
        raise ValueError('Cannot apply cumax to a tensor that is 1D')
    return K.cumsum(softmax(x, axis), axis)
