# Keras Ordered Neurons LSTM

[![Version](https://img.shields.io/pypi/v/keras-ordered-neurons.svg)](https://pypi.org/project/keras-ordered-neurons/)

\[[中文](https://github.com/CyberZHG/keras-ordered-neurons/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-ordered-neurons/blob/master/README.md)\]

[ON-LSTM](https://openreview.net/pdf?id=B1l6qiR5F7)的非官方实现。

## 安装

```bash
pip install keras-ordered-neurons
```

## 使用

### 基本

使用起来和`LSTM`基本一致，默认情况下还需要一个`chunk_size`参数，代表master gates缩小的倍数：

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

参数中的`recurrent_dropconnect`用于设置隐藏状态权重矩阵的随机归零概率：

```python
from keras_ordered_neurons import ONLSTM

ONLSTM(units=50, chunk_size=5, recurrent_dropconnect=0.2)
```

### 获取期望分割点

将`return_splits`设置为`True`来返回master forget gate和master input gate的期望分割点：

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
