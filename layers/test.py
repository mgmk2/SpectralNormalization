import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from spectral_normalization_core import SNDense

parser = argparse.ArgumentParser()
parser.add_argument('--use_tpu', action='store_true')
pargs = parser.parse_args()

def get_session(tpu_cluster_resolver=None):
    if pargs.use_tpu and tpu_cluster_resolver is not None:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        cluster_spec = tpu_cluster_resolver.cluster_spec()
        if cluster_spec:
            config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        return tf.Session(
            target=tpu_cluster_resolver.master(),
            config=config)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    return tf.Session(config=config)


def tpu_decorator(func):
    def wrapper(*args, **kwargs):
        tf.keras.backend.clear_session()
        if pargs.use_tpu:
            tpu_grpc_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
            tf.contrib.distribute.initialize_tpu_system(tpu_cluster_resolver)
            strategy = tf.contrib.distribute.TPUStrategy(tpu_cluster_resolver)
            kwargs['tpu_cluster_resolver'] = tpu_cluster_resolver
            with strategy.scope():
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def power_iteration(W, u, power_iter=1):
    W_T = tf.transpose(W)
    for i in range(power_iter):
        v = tf.nn.l2_normalize(tf.matmul(u, W))  # 1 x filters
        u = tf.nn.l2_normalize(tf.matmul(v, W_T))
    v_bar = tf.stop_gradient(v)
    u_bar = tf.stop_gradient(u)
    sigma_W = tf.squeeze(tf.matmul(tf.matmul(u_bar, W), tf.transpose(v_bar)))
    return u_bar, sigma_W, v_bar

def build_model(input_dim, output_dim):
    inputs = Input((input_dim,))
    outputs = SNDense(output_dim)(inputs)
    model = Model(inputs, outputs)
    return model

@tpu_decorator
def sv_estimation_test(input_dim, output_dim, tpu_cluster_resolver=None):
    model = build_model(input_dim, output_dim)
    W = model.layers[-1].kernel
    u = model.layers[-1].u
    x = tf.ones((1, input_dim), tf.float32)
    y = model(x)
    _, sigma_W, _ = power_iteration(W, u, power_iter=1)
    _, sigma_W_100, _ = power_iteration(W, u, power_iter=100)

    with get_session(tpu_cluster_resolver=tpu_cluster_resolver) as sess:
        sess.run(tf.global_variables_initializer())
        s = sess.run(sigma_W)
        s_100 = sess.run(sigma_W_100)

        W_np = sess.run(W)
        _u, sv, _vh = np.linalg.svd(W_np)

        print('[singular value estimation test]')
        print('power iteration with tensorflow')
        print('      1 iteration  : {:}'.format(s))
        print('    100 iteration  : {:}'.format(s_100))
        print('estimation by numpy: {:}'.format(sv[0]))
        print()

@tpu_decorator
def grad_test(input_dim, output_dim, tpu_cluster_resolver=None):
    model = build_model(input_dim, output_dim)
    x = tf.ones((1, input_dim), tf.float32)
    y = model(x)
    W = model.layers[-1].kernel
    u = model.layers[-1].u
    u_bar, sigma_W, v_bar = power_iteration(W, u, power_iter=1)
    W_bar = W / sigma_W

    grad = tf.gradients(ys=W_bar, xs=W)

    uv = tf.matmul(tf.transpose(u_bar), v_bar)
    grad_sigma_inv = -uv / sigma_W ** 2
    grad2 = 1 / sigma_W + grad_sigma_inv * tf.reduce_sum(W)

    with get_session(tpu_cluster_resolver=tpu_cluster_resolver) as sess:
        sess.run(tf.global_variables_initializer())
        g, = sess.run(grad)
        g2 = sess.run(grad2)
        print('[gradients calculation test]')
        print('gradients calculated by tf.gradients()')
        print(g)
        print('gradients calculated by exact solution')
        print(g2)
        print()

@tpu_decorator
def sv_assign_test(input_dim, output_dim, training=True, tpu_cluster_resolver=None):
    model = build_model(input_dim, output_dim)
    W = model.layers[-1].kernel
    u = model.layers[-1].u
    x = tf.ones((1, input_dim), tf.float32)
    y = model(x, training=training)
    update_ops = model.updates
    u_power_iter, _, _ = power_iteration(W, u, power_iter=1)

    with get_session(tpu_cluster_resolver=tpu_cluster_resolver) as sess:
        sess.run(tf.global_variables_initializer())
        u0 = sess.run(u)
        u1_power_iter = sess.run(u_power_iter)
        sess.run(update_ops)
        u1 = sess.run(u)

        print('[singular vector assign test when training = ' + str(training) + ']')
        print('initial singular vector')
        print(u0)
        print('assigned singular vector')
        print(u1)
        print('singular vector calculated outside of Model')
        print(u1_power_iter)
        print()

if __name__ == '__main__':
    input_dim = 10
    output_dim = 2
    print('input dim: 10')
    print('output dim: 2')

    sv_estimation_test(input_dim, output_dim)
    grad_test(input_dim, output_dim)
    sv_assign_test(input_dim, output_dim, training=True)
    sv_assign_test(input_dim, output_dim, training=False)
