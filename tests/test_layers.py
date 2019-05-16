import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_ordered_neurons.backend import models, layers, callbacks
from keras_ordered_neurons.backend import backend as K
from keras_ordered_neurons import ONLSTM


class TestONLSTM(TestCase):

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            model = models.Sequential()
            model.add(ONLSTM(units=13, chunk_size=5, input_shape=(None, 100)))

    def test_return_all_splits(self):
        if K.backend() == 'cntk':
            return
        inputs = layers.Input(shape=(None,))
        embed = layers.Embedding(input_dim=10, output_dim=100)(inputs)
        outputs = ONLSTM(units=50, chunk_size=5, return_sequences=True, return_splits=True)(embed)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        model.summary(line_length=120)
        predicted = model.predict(np.random.randint(0, 10, (3, 7)))
        self.assertEqual((3, 7, 50), predicted[0].shape)
        self.assertEqual((3, 7, 2), predicted[1].shape)

    def test_return_last_splits(self):
        if K.backend() == 'cntk':
            return
        inputs = layers.Input(shape=(None,))
        embed = layers.Embedding(input_dim=10, output_dim=100)(inputs)
        outputs = ONLSTM(units=50, chunk_size=5, return_splits=True)(embed)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        model.summary(line_length=120)
        predicted = model.predict(np.random.randint(0, 10, (3, 7)))
        self.assertEqual((3, 50), predicted[0].shape)
        self.assertEqual((3, 2), predicted[1].shape)

    def test_fit_classification(self):
        model = models.Sequential()
        model.add(layers.Embedding(input_shape=(None,), input_dim=10, output_dim=100, mask_zero=True))
        model.add(layers.Bidirectional(ONLSTM(
            units=50,
            chunk_size=5,
            dropout=0.1,
            recurrent_dropconnect=0.1,
            use_bias=False,
            return_sequences=True,
        )))
        model.add(layers.Bidirectional(ONLSTM(
            units=50,
            chunk_size=5,
            recurrent_dropout=0.1,
            return_sequences=True,
        )))
        model.add(layers.Bidirectional(ONLSTM(units=50, chunk_size=5, unit_forget_bias=False)))
        model.add(layers.Dense(units=2, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        model_path = os.path.join(tempfile.gettempdir(), 'test_on_lstm_%f.h5' % np.random.random())
        model.save(model_path)
        model = models.load_model(model_path, custom_objects={'ONLSTM': ONLSTM})

        data_size, seq_len = 10000, 17
        x = np.random.randint(0, 10, (data_size, seq_len))
        y = [0] * data_size
        for i in range(data_size):
            if 3 in x[i].tolist() and 7 in x[i].tolist():
                y[i] = 1
        y = np.array(y)
        model.summary()
        model.fit(
            x,
            y,
            epochs=10,
            callbacks=[callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=2)],
        )

        model_path = os.path.join(tempfile.gettempdir(), 'test_on_lstm_%f.h5' % np.random.random())
        model.save(model_path)
        model = models.load_model(model_path, custom_objects={'ONLSTM': ONLSTM})

        predicted = model.predict(x).argmax(axis=-1)
        self.assertLess(np.sum(np.abs(y - predicted)), data_size // 100)
