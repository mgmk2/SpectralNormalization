# SpectralNormalization
Implementation of Spectral Normalization for TensorFlow keras API.
This layers are available on Distribute Strategy (ex. TPU).

This is a modification of TensorFlow keras layers and code is derived from [TensorFlow ver1.14](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow).

# Features
TensorFlow keras layers class with spectral normalization:
* SNDense
* SNConv1D
* SNConv2D
* SNConv3D

# Requirement
tensorflow >= 1.14

# Usage
Replace keras layers (Conv2D etc) with layers using spectral normalization as below:
```python
from SpectralNormalization.layers import SNConv2D

outputs = SNConv2D(64, (3, 3), padding='same')(inputs)
```

You can set `singular_vector_initializer` and `power_iter` arguments, which affect singular value estimation in spectral normalization.
And also, you can use the same arguments as original keras layers.
