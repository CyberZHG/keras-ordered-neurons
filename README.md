# Keras Ordered Neurons LSTM

[![Version](https://img.shields.io/pypi/v/keras-ordered-neurons.svg)](https://pypi.org/project/keras-ordered-neurons/)

\[[中文](https://github.com/CyberZHG/keras-ordered-neurons/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-ordered-neurons/blob/master/README.md)\]

Unofficial implementation of [ON-LSTM](https://openreview.net/pdf?id=B1l6qiR5F7).

## Install

```bash
pip install keras-ordered-neurons
```

## Usage

### Basic

Same as `LSTM` except that an extra argument `chunk_size` should be given:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense

from keras_ordered_neurons import ONLSTM

model = Sequential()
model.add(Embedding(input_shape=(None,), input_dim=10, output_dim=100))
model.add(Bidirectional(ONLSTM(units=50, chunk_size=5)))
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()
```

### DropConnect

Set `recurrent_dropconnect` to a non-zero value to enable drop-connect for recurrent weights:

```python
from keras_ordered_neurons import ONLSTM

ONLSTM(units=50, chunk_size=5, recurrent_dropconnect=0.2)
```

### Expected Split Points

Set `return_splits` to `True` if you want to know the expected split points of master forget gate and master input gate.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding

from keras_ordered_neurons import ONLSTM

inputs = Input(shape=(None,))
embed = Embedding(input_dim=10, output_dim=100)(inputs)
outputs, splits = ONLSTM(units=50, chunk_size=5, return_sequences=True, return_splits=True)(embed)
model = Model(inputs=inputs, outputs=splits)
model.compile(optimizer='adam', loss='mse')
model.summary(line_length=120)
```
