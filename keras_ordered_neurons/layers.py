import warnings

from .backend import layers, activations, initializers, regularizers, constraints
from .backend import backend as K

from .activations import cumax

__all__ = ['ONLSTMCell', 'ONLSTM']


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(dropped_inputs, ones, training=training) for _ in range(count)]
    return K.in_train_phase(dropped_inputs, ones, training=training)


class ONLSTMCell(layers.Layer):
    """Cell class for the ON-LSTM layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        chunk_size: Chunk size of the master gates.
        activation: Activation function to use
            Default: hyperbolic tangent (`tanh`).
        recurrent_activation: Activation function to use for the recurrent step
            Default: hard sigmoid (`hard_sigmoid`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the recurrent state.
    """

    def __init__(self, units,
                 chunk_size,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 recurrent_dropconnect=0.,
                 return_splits=False,
                 **kwargs):
        super(ONLSTMCell, self).__init__(**kwargs)
        if units % chunk_size != 0:
            raise ValueError('`units` should be divisible by `chunk_size`, found: {} and {}.'.format(units, chunk_size))
        self.units = units
        self.chunk_size = chunk_size
        self.master_units = self.units // self.chunk_size
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.recurrent_dropconnect = min(1., max(0., recurrent_dropconnect))

        self.return_splits = return_splits

        if return_splits:
            self.state_size = (self.units + 2, self.units)
        else:
            self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self._recurrect_dropconnect_masks = None

        self.kernel, self.recurrent_kernel, self.bias = None, None, None
        self.kernel_i, self.kernel_f, self.kernel_c, self.kernel_o, self.kernel_mf, self.kernel_mi = (None,) * 6
        self.recurrent_kernel_i, self.recurrent_kernel_f = None, None
        self.recurrent_kernel_c, self.recurrent_kernel_o = None, None
        self.recurrent_kernel_mf, self.recurrent_kernel_mi = None, None
        self.bias_i, self.bias_f, self.bias_c, self.bias_o, self.bias_mf, self.bias_mi = (None,) * 6

    def build(self, input_shape):
        input_dim = input_shape[-1]
        output_dim = self.units * 4 + self.master_units * 2

        self.kernel = self.add_weight(
            shape=(input_dim, output_dim),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, output_dim),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2 + self.master_units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(output_dim,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units:self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2:self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:self.units * 4]
        self.kernel_mf = self.kernel[:, self.units * 4:self.units * 4 + self.master_units]
        self.kernel_mi = self.kernel[:, self.units * 4 + self.master_units:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units:self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2:self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:self.units * 4]
        self.recurrent_kernel_mf = self.recurrent_kernel[:, self.units * 4:self.units * 4 + self.master_units]
        self.recurrent_kernel_mi = self.recurrent_kernel[:, self.units * 4 + self.master_units:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units:self.units * 2]
            self.bias_c = self.bias[self.units * 2:self.units * 3]
            self.bias_o = self.bias[self.units * 3:self.units * 4]
            self.bias_mf = self.bias[self.units * 4:self.units * 4 + self.master_units]
            self.bias_mi = self.bias[self.units * 4 + self.master_units:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
            self.bias_mf = None
            self.bias_mi = None
        super(ONLSTMCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        if self.return_splits:
            h_tm1 = h_tm1[:, :-2]

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=6)
        if 0 < self.recurrent_dropout < 1 and self._recurrent_dropout_mask is None:
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=6)

        if 0 < self.recurrent_dropconnect < 1 and self._recurrect_dropconnect_masks is None:
            self._recurrect_dropconnect_masks = [
                _generate_dropout_mask(
                    K.ones_like(recurrent_kernel),
                    self.recurrent_dropconnect,
                    training=training,
                ) for recurrent_kernel in [
                    self.recurrent_kernel_i,
                    self.recurrent_kernel_f,
                    self.recurrent_kernel_c,
                    self.recurrent_kernel_o,
                    self.recurrent_kernel_mf,
                    self.recurrent_kernel_mi,
                ]]

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask
        # dropconnect matrices for recurrent weights
        rec_dc_mask = self._recurrect_dropconnect_masks

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
            inputs_mf = inputs * dp_mask[4]
            inputs_mi = inputs * dp_mask[5]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
            inputs_mf = inputs
            inputs_mi = inputs
        x_i = K.dot(inputs_i, self.kernel_i)
        x_f = K.dot(inputs_f, self.kernel_f)
        x_c = K.dot(inputs_c, self.kernel_c)
        x_o = K.dot(inputs_o, self.kernel_o)
        x_mf = K.dot(inputs_mf, self.kernel_mf)
        x_mi = K.dot(inputs_mi, self.kernel_mi)
        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            x_f = K.bias_add(x_f, self.bias_f)
            x_c = K.bias_add(x_c, self.bias_c)
            x_o = K.bias_add(x_o, self.bias_o)
            x_mf = K.bias_add(x_mf, self.bias_mf)
            x_mi = K.bias_add(x_mi, self.bias_mi)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
            h_tm1_mf = h_tm1 * rec_dp_mask[4]
            h_tm1_mi = h_tm1 * rec_dp_mask[5]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
            h_tm1_mf = h_tm1
            h_tm1_mi = h_tm1

        if 0 < self.recurrent_dropconnect < 1.:
            recurrent_kernel_i = self.recurrent_kernel_i * rec_dc_mask[0]
            recurrent_kernel_f = self.recurrent_kernel_f * rec_dc_mask[1]
            recurrent_kernel_c = self.recurrent_kernel_c * rec_dc_mask[2]
            recurrent_kernel_o = self.recurrent_kernel_o * rec_dc_mask[3]
            recurrent_kernel_mf = self.recurrent_kernel_mf * rec_dc_mask[4]
            recurrent_kernel_mi = self.recurrent_kernel_mi * rec_dc_mask[5]
        else:
            recurrent_kernel_i = self.recurrent_kernel_i
            recurrent_kernel_f = self.recurrent_kernel_f
            recurrent_kernel_c = self.recurrent_kernel_c
            recurrent_kernel_o = self.recurrent_kernel_o
            recurrent_kernel_mf = self.recurrent_kernel_mf
            recurrent_kernel_mi = self.recurrent_kernel_mi

        f = self.recurrent_activation(x_f + K.dot(h_tm1_f, recurrent_kernel_f))
        i = self.recurrent_activation(x_i + K.dot(h_tm1_i, recurrent_kernel_i))
        mf = cumax(x_mf + K.dot(h_tm1_mf, recurrent_kernel_mf))
        mi = 1.0 - cumax(x_mi + K.dot(h_tm1_mi, recurrent_kernel_mi))
        if self.return_splits:
            df = self.master_units - K.sum(mf, axis=-1, keepdims=True)
            di = K.sum(mi, axis=-1, keepdims=True)
        mf = K.repeat_elements(mf, self.chunk_size, axis=-1)
        mi = K.repeat_elements(mi, self.chunk_size, axis=-1)
        w = mf * mi
        f = f * w + (mf - w)
        i = i * w + (mi - w)
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c, recurrent_kernel_c))
        o = self.recurrent_activation(x_o + K.dot(h_tm1_o, recurrent_kernel_o))

        h = o * self.activation(c)
        if self.return_splits:
            h = K.concatenate([h, df, di], axis=-1)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {
            'units': self.units,
            'chunk_size': self.chunk_size,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'unit_forget_bias': self.unit_forget_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'recurrent_dropconnect': self.recurrent_dropconnect,
            'return_splits': self.return_splits,
        }
        base_config = super(ONLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ONLSTM(layers.RNN):
    """Ordered Neurons LSTM

    # Arguments
        units: Positive integer, dimensionality of the output space.
        chunk_size: Chunk size of the master gates.
        activation: Activation function to use
            Default: hyperbolic tangent (`tanh`).
        recurrent_activation: Activation function to use
            for the recurrent step
            Default: hard sigmoid (`hard_sigmoid`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the recurrent state.
        recurrent_dropconnect: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of hidden states.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks]
          (https://openreview.net/pdf?id=B1l6qiR5F7)
    """

    def __init__(self, units,
                 chunk_size,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 recurrent_dropconnect=0.,
                 return_sequences=False,
                 return_state=False,
                 return_splits=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
            recurrent_dropconnect = 0.
        cell = ONLSTMCell(
            units,
            chunk_size,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            recurrent_dropconnect=recurrent_dropconnect,
            return_splits=return_splits,
        )
        self.return_splits = return_splits
        super(ONLSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def compute_output_shape(self, input_shape):
        output_shapes = super(ONLSTM, self).compute_output_shape(input_shape)
        if self.return_splits:
            if not isinstance(output_shapes, list):
                output_shapes = [output_shapes]
            if self.return_sequences:
                output_shapes.append((input_shape[0], input_shape[1], 2))
            else:
                output_shapes.append((input_shape[0], 2))
        return output_shapes

    def compute_mask(self, inputs, mask):
        outputs_masks = super(ONLSTM, self).compute_mask(inputs, mask)
        if self.return_splits:
            if not isinstance(outputs_masks, list):
                outputs_masks = [outputs_masks]
            outputs_masks.append(None)
        return outputs_masks

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        outputs = super(ONLSTM, self).call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
        )
        if self.return_splits:
            if not isinstance(outputs, list):
                outputs = [outputs]
            if self.return_sequences:
                splits = outputs[0][:, :, -2:]
                outputs[0] = outputs[0][:, :, :-2]
            else:
                splits = outputs[0][:, -2:]
                outputs[0] = outputs[0][:, :-2]
            outputs.append(splits)
        return outputs

    @property
    def units(self):
        return self.cell.units

    @property
    def chunk_size(self):
        return self.cell.chunk_size

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def recurrent_dropconnect(self):
        return self.cell.recurrent_dropconnect

    def get_config(self):
        config = {
            'units': self.units,
            'chunk_size': self.chunk_size,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'unit_forget_bias': self.unit_forget_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'recurrent_dropconnect': self.recurrent_dropconnect,
            'return_splits': self.return_splits,
        }
        base_config = super(ONLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
