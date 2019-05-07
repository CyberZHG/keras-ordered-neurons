from unittest import TestCase
import numpy as np
from keras_ordered_neurons.backend import backend as K
from keras_ordered_neurons import cumax


class TestCumax(TestCase):

    def test_invalid(self):
        x = K.placeholder(ndim=1)

        with self.assertRaises(ValueError):
            f = K.function([x], [cumax(x)])

    def test_valid_default_axis(self):
        x = K.placeholder(shape=(3, 4))
        f = K.function([x], [cumax(x)])
        predicted = f([np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ])])[0]
        self.assertTrue(np.allclose(
            np.array([0.25, 0.5, 0.75, 1.0]),
            predicted[0]
        ), predicted)
