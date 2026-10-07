"""Reconstructed U-Net builder inspired by thesis section 3.3, not recovered code.

New choices: three pooling levels, two 3x3 convolutions/block, concatenated
skips and transposed-convolution upsampling. The thesis describes filters
64→512, but its final-channel and parameter-count descriptions are ambiguous.
Callers must explicitly select output channels and activation; pretrained
checkpoint compatibility is NOT implied. No weights are shipped or downloaded.
"""

def build_unet(*, output_channels: int, output_activation: str):
    """Build an untrained 256x256 RGB reference model with an explicit head.

    Use (1, 'sigmoid') only for binary labels; (3, 'softmax') requires a verified
    three-class mask schema. Other positive channel counts are supported.
    TensorFlow is optional until this function is called.
    """
    if not isinstance(output_channels, int) or output_channels < 1:
        raise ValueError("output_channels must be a positive integer")
    if output_activation not in {"sigmoid", "softmax"}:
        raise ValueError("Choose sigmoid or softmax explicitly")
    if output_activation == "softmax" and output_channels == 1:
        raise ValueError("Single-channel softmax is not meaningful")
    from tensorflow import keras
    L = keras.layers
    def block(x, filters):
        x = L.Conv2D(filters, 3, activation="relu", padding="same")(x)
        return L.Conv2D(filters, 3, activation="relu", padding="same")(x)
    inputs = keras.Input((256, 256, 3))
    x, skips = inputs, []
    for width in (64, 128, 256):
        x = block(x, width); skips.append(x); x = L.MaxPooling2D(2)(x)
    x = block(x, 512)
    for width, skip in zip((256, 128, 64), reversed(skips)):
        x = L.Conv2DTranspose(width, 2, strides=2, padding="same")(x)
        x = block(L.Concatenate()([x, skip]), width)
    output = L.Conv2D(output_channels, 1, activation=output_activation)(x)
    return keras.Model(inputs, output, name="reconstructed_unet")


def dice_loss(y_true, y_pred):
    """New soft-Dice loss: average per-example overlap across spatial/channels.

    Smoothing 1e-6 and reduction axes are reconstruction choices, not recovered
    original settings. Ensure predictions and target masks share a schema.
    """
    import tensorflow as tf
    a, b = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    tf.debugging.assert_equal(tf.shape(a), tf.shape(b))
    intersection = tf.reduce_sum(a*b, axis=(1,2,3))
    denominator = tf.reduce_sum(a+b, axis=(1,2,3))
    return 1-tf.reduce_mean((2*intersection+1e-6)/(denominator+1e-6))
