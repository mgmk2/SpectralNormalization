# Code derived from
# https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/layers/core.py
# and
# https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/layers/normalization.py

from tensorflow.python.distribute import distribution_strategy_context

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
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
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables

from tensorflow.python.keras.layers.core import Dense

class SNDense(Dense):
    """Just your regular densely-connected NN layer with spectral normalization.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Example:

    ```python
    # as first layer in a sequential model:
    model = Sequential()
    model.add(Dense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)

    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(Dense(32))
    ```

    Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    singular_vector_initializer: Initializer for
        the singular vector used in spectral normalization.
    power_iter: Positive integer,
        number of iteration in singular value estimation.

    Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
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

        super(SNDense, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.singular_vector_initializer = singular_vector_initializer
        self.power_iter = power_iter

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.u = self.add_weight(
            name='singular_vector',
            shape=(1, last_dim),
            initializer=self.singular_vector_initializer,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype)
        super(SNDense, self).build(input_shape)

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
        W = self.kernel
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
                            self._assign_singular_vector, (u_bar,),
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

        # normal Dense using W_bar
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, W_bar, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            # Cast the inputs to self.dtype, which is the variable dtype. We do not
            # cast if `should_cast_variables` is True, as in that case the variable
            # will be automatically casted to inputs.dtype.
            if not self._mixed_precision_policy.should_cast_variables:
                inputs = math_ops.cast(inputs, self.dtype)
            outputs = gen_math_ops.mat_mul(inputs, W_bar)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def get_config(self):
        config = {
            'singular_vector_initializer': initializers.serialize(
                                           self.singular_vector_initializer),
            'power_iter': self.power_iter
        }
        base_config = super(SNDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
