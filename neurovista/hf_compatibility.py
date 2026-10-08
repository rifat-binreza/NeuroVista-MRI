"""Reference adapter for Gradio 5.37 Image(type='filepath', image_mode='RGB').

Labels and preprocessing follow the original Brain_Tumor_MRI_Detection app.
No weights, downloads, deployment code or private settings are included.
Matching software behavior does not establish diagnostic accuracy.
"""
import io
import numpy as np
from PIL import Image, ImageOps

REFERENCE_LABELS = ('Glioma', 'Meningioma', 'No Tumor', 'Pituitary')


def prepare_reference_input(path, task='classification'):
    """Return float32 NHWC input using the reference's shared upload rules.

    RGB files pass through. Other modes convert to RGB; grayscale JPEGs receive
    a second JPEG encode with Pillow defaults. PNG conversion is lossless.
    No filename-specific rules or score adjustments are used.
    """
    if task not in ('classification', 'segmentation'):
        raise ValueError('Choose classification or segmentation')
    size = 299 if task == 'classification' else 256
    with Image.open(path) as source:
        if source.format not in ('JPEG', 'PNG'):
            raise ValueError('Use JPEG or PNG')
        if source.width * source.height > 16_000_000 or getattr(source, 'n_frames', 1) != 1:
            raise ValueError('Use a single still image under 16 megapixels')
        source.load()
        if source.mode == 'RGB':
            # Gradio returns the original filepath here, without EXIF transpose.
            image = source.copy()
        else:
            image = ImageOps.exif_transpose(source).convert('RGB')
            if source.format == 'JPEG':
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG')
                buffer.seek(0)
                with Image.open(buffer) as converted:
                    converted.load()
                    image = converted.convert('RGB')
        array = np.asarray(image.resize((size, size), Image.Resampling.NEAREST), dtype=np.float32)
    return array[None, ...] / 255.0


def classify_reference(path, model):
    """Classify with a caller-supplied trusted checkpoint and reference labels.

    Only use this order for the reference checkpoints. These are uncalibrated
    model scores; private input-screening and abstention policies are separate.
    """
    scores = np.asarray(model.predict(prepare_reference_input(path), verbose=0))
    if scores.shape != (1, 4) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid classifier output')
    if not np.isclose(scores.sum(), 1.0, atol=0.02):
        raise ValueError('Expected normalized class scores')
    return dict(zip(REFERENCE_LABELS, (float(score) for score in scores[0])))
