"""Local classification adapter for user-supplied trusted Keras checkpoints.

This selective public example never downloads weights and does not embed
private model links. It is independent of the private web-serving system.
"""
import numpy as np
from .preprocessing import prepare_image


def classify(image_path, model, *, labels):
    """Return scores using a caller-verified class order and loaded model.

    This is a model score, not a diagnosis or calibrated disease probability.
    """
    labels = tuple(labels)
    if len(labels)!=4 or len(set(labels))!=4:
        raise ValueError("Supply exactly four distinct verified class names")
    scores = np.asarray(model.predict(prepare_image(image_path,"classification"), verbose=0))
    if scores.shape!=(1,4) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError("Invalid classifier output")
    if not np.isclose(scores.sum(),1.,atol=.02):
        raise ValueError("Expected softmax class scores")
    return {label: float(value) for label,value in zip(labels,scores[0])}
