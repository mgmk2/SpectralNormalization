# Code derived from
# https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/layers/convolutional.py
# and
# https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/layers/normalization.py

from functools import reduce
from operator import mul

from tensorflow.python.distribute import distribution_strategy_context

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables

from tensorflow.python.keras.layers.convolutional import Conv

class SNConv(Conv):
    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 singular_vector_initializer=initializers.RandomNormal(0, 1),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 power_iter=1,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(SNConv, self).__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            trainable=trainable,
            name=name,
            **kwargs)
        self.singular_vector_initializer = singular_vector_initializer
        self.power_iter = power_iter

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        singular_vector_shape = (1, reduce(mul, self.kernel_size) * input_dim)

        self.u = self.add_weight(
            name='singular_vector',
            shape=singular_vector_shape,
            initializer=self.singular_vector_initializer,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype)
        super(SNConv, self).build(input_shape)

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if base_layer_utils.is_in_keras_graph():
            training = math_ops.logical_and(training, self._get_trainable_var())
        else:
            training = math_ops.logical_and(training, self.trainable)
        return training

    def _assign_singular_vector(self, variable, value):
        with K.name_scope('AssignSingularVector') as scope:
            with ops.colocate_with(variable):
                return state_ops.assign(variable, value, name=scope)

    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        # Update singular vector by power iteration
        if self.data_format == 'channels_first':
            W_T = array_ops.reshape(self.kernel, (self.filters, -1))
            W = array_ops.transpose(W_T)
        else:
            W = array_ops.reshape(self.kernel, (-1, self.filters))
            W_T = array_ops.transpose(W)
        u = self.u
        for i in range(self.power_iter):
            v = nn_impl.l2_normalize(math_ops.matmul(u, W))  # 1 x filters
            u = nn_impl.l2_normalize(math_ops.matmul(v, W_T))
        # Backprop doesn't need in power iteration
        u_bar = gen_array_ops.stop_gradient(u)
        v_bar = gen_array_ops.stop_gradient(v)
        # Spectral Normalization
        sigma_W = math_ops.matmul(math_ops.matmul(u_bar, W), array_ops.transpose(v_bar))
        W_bar = self.kernel / array_ops.squeeze(sigma_W)

        # Assign new singular vector
        training_value = tf_utils.constant_value(training)
        if training_value is not False:
            if distribution_strategy_context.in_cross_replica_context():
                strategy = distribution_strategy_context.get_strategy()

                def u_update():
                    def true_branch():
                        return strategy.extended.update(
                            self.u,
                            self._assign_in_strategy, (u_bar,),
                            group=False)
                    def false_branch():
                        return strategy.unwrap(self.u)
                    return tf_utils.smart_cond(training, true_branch, false_branch)
            else:
                def u_update():
                    def true_branch():
                        return self._assign_singular_vector(self.u, u_bar)
                    def false_branch():
                        return self.u
                    return tf_utils.smart_cond(training, true_branch, false_branch)
            self.add_update(u_update, inputs=True)

        # normal convolution using W_bar
        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
          if self.data_format == 'channels_first':
            if self.rank == 1:
              # nn.bias_add does not accept a 1D input tensor.
              bias = array_ops.reshape(self.bias, (1, self.filters, 1))
              outputs += bias
            else:
              outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
          else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
          return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'singular_vector_initializer': initializers.serialize(
                                           self.singular_vector_initializer),
            'power_iter': self.power_iter
        }
        base_config = super(SNConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SNConv1D(SNConv):

    def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               singular_vector_initializer=initializers.RandomNormal(0, 1),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               power_iter=1,
               **kwargs):
        super(SNConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            singular_vector_initializer=initializers.get(singular_vector_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            power_iter=power_iter,
            **kwargs)

    def call(self, inputs):
        if self.padding == 'causal':
          inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(SNConv1D, self).call(inputs)


class SNConv2D(SNConv):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 singular_vector_initializer=initializers.RandomNormal(0, 1),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 power_iter=1,
                 **kwargs):
        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            singular_vector_initializer=initializers.get(singular_vector_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            power_iter=power_iter,
            **kwargs)


class SNConv3D(SNConv):
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               singular_vector_initializer=initializers.RandomNormal(0, 1),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               power_iter=1,
               **kwargs):
        super(SNConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            singular_vector_initializer=initializers.get(singular_vector_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            power_iter=power_iter,
            **kwargs)
