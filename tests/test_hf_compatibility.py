"""Adapter regression checks without TensorFlow or checkpoints."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from neurovista.hf_compatibility import prepare_reference_input, classify_reference


class CompatibilityTests(unittest.TestCase):
    def test_rgb_pixels_pass_through(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rgb.png'
            pixels = np.arange(256*256*3, dtype=np.uint8).reshape(256,256,3)
            Image.fromarray(pixels).save(path)
            np.testing.assert_array_equal(prepare_reference_input(path, 'segmentation'), pixels.astype(np.float32)[None]/255)

    def test_grayscale_png_is_lossless(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gray.png'
            pixels = np.tile(np.arange(256, dtype=np.uint8), (256,1))
            Image.fromarray(pixels).save(path)
            expected = np.repeat(pixels[...,None], 3, axis=2).astype(np.float32)[None]/255
            np.testing.assert_array_equal(prepare_reference_input(path, 'segmentation'), expected)

    def test_label_indices(self):
        class Model:
            def predict(self, image, verbose=0):
                return np.array([[0, .01, .98, .01]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            Image.new('L', (299,299)).save(path)
            result = classify_reference(path, Model())
            self.assertEqual(result['No Tumor'], .98)
            self.assertEqual(result['Pituitary'], .01)
