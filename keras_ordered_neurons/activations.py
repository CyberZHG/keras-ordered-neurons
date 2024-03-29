from tensorflow import keras
from tensorflow.keras import backend as K


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
    return K.cumsum(keras.activations.softmax(x, axis), axis)
