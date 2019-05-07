from unittest import TestCase
import numpy as np
from keras_ordered_neurons.backend import backend as K


class TestCumsum(TestCase):

    def test_valid_last_axis(self):
        if K.backend() != 'cntk':
            return
        from keras_ordered_neurons.cntk_backend import cumsum
        x = K.placeholder(shape=(3, 4))
        f = K.function([x], [cumsum(x, axis=-1)])
        predicted = f([np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ])])[0]
        self.assertTrue(np.allclose(
            np.array([
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 3.0, 6.0, 10.0],
                [4.0, 7.0, 9.0, 10.0],
            ]),
            predicted
        ), predicted)

    def test_invalid_axis(self):
        if K.backend() != 'cntk':
            return
        from keras_ordered_neurons.cntk_backend import cumsum
        with self.assertRaises(ValueError):
            x = K.placeholder(shape=(1, 3, 4))
            f = K.function([x], [cumsum(x, axis=1)])
