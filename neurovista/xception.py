"""Thesis-informed Xception reference builder (section 3.6).

Flatten → Dropout(0.5) → Dense(128, ReLU) → Dropout(0.5) → Dense(4,
softmax) is a reconstruction of the prose. Exact dropout placement and base
freezing schedule are not recoverable. No trained tumor model is provided.
"""

def build_xception(*, base_trainable: bool, imagenet_weights=False):
    """Build an untrained tumor classifier; downloading ImageNet is opt-in.

    Thesis/notebook inputs use /255. Keras's standard Xception preprocessing
    instead maps to [-1,1]; changing that convention changes the experiment.
    Decide and record preprocessing before training; never switch it silently
    when loading an existing checkpoint.
    """
    from tensorflow import keras
    base = keras.applications.Xception(include_top=False,
        weights="imagenet" if imagenet_weights else None, input_shape=(299,299,3))
    base.trainable = base_trainable
    x = keras.layers.Flatten()(base.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(base.input, outputs, name="reconstructed_xception")
