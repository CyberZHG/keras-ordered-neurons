# Keras Ordered Neurons LSTM

[![Travis](https://travis-ci.org/CyberZHG/keras-ordered-neurons.svg)](https://travis-ci.org/CyberZHG/keras-ordered-neurons)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-ordered-neurons/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-ordered-neurons)
[![996.ICU](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://996.icu) 

Unofficial implementation of [ON-LSTM](https://openreview.net/pdf?id=B1l6qiR5F7).

## Install

```bash
pip install keras-ordered-neurons
```

## Usage

### Basic

Same as `LSTM` except that an extra argument `chunk_size` should be given:

```python
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, Dense
from keras_ordered_neurons import ONLSTM

model = Sequential()
model.add(Embedding(input_shape=(None,), input_dim=10, output_dim=100))
model.add(Bidirectional(ONLSTM(units=50, chunk_size=5)))
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()
```

### Expected Split Points

Set `return_splits` to `True` if you want to know the expected split points of master forget gate and master input gate.

```python
from keras.models import Model
from keras.layers import Input, Embedding
from keras_ordered_neurons import ONLSTM

inputs = Input(shape=(None,))
embed = Embedding(input_dim=10, output_dim=100)(inputs)
outputs, splits = ONLSTM(units=50, chunk_size=5, return_sequences=True, return_splits=True)(embed)
model = Model(inputs=inputs, outputs=splits)
model.compile(optimizer='adam', loss='mse')
model.summary(line_length=120)
```

### `tf.keras`

Add `TF_KERAS=1` to environment variables if you are using `tensorflow.python.keras`.
