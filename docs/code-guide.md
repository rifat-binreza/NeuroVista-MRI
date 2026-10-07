# Working with the reference modules

Run commands from the repository root, using Python 3.11 or newer. The standard dependencies support preprocessing, metrics, planning and tests. TensorFlow is imported lazily by the model builders; install a compatible TensorFlow runtime separately. No private weights or dataset URLs are provided.

## Preprocess a permitted image
```python
from neurovista.preprocessing import prepare_image
x = prepare_image("local_mri.png", "classification")  # (1, 299, 299, 3), float32
```

## Build reference models
```python
from neurovista.unet import build_unet, dice_loss
from neurovista.xception import build_xception
seg = build_unet(output_channels=1, output_activation="sigmoid")
cls = build_xception(base_trainable=False, imagenet_weights=False)
```
These models have randomly initialized weights by default and must not be used for meaningful predictions. The thesis reports ImageNet initialization for Xception; `imagenet_weights=True` explicitly permits that download. Freezing the base here is a new starting choice, not a recovered training schedule. The model builders have been syntax-checked but have not been executed with TensorFlow in this workspace.

## Evaluate actual predictions
```python
from neurovista.metrics import binary_overlap, classification_report
print(binary_overlap([1, 1, 0, 0], [1, 0, 1, 0]))
# Known arithmetic fixture: Dice 0.5, IoU 1/3 — not a research benchmark.
```
Pass one mask at a time, then choose and document patient-level aggregation. Classification metrics take integer class IDs and an explicit class-name order. Missing-class ratios are zero; both-empty binary masks score one by convention.

## Plan experiments
`python -m neurovista.experiments` emits the 21-condition plan as JSON. Real counts refer to unique retained samples; oversampling does not create new patients. Fix patient-level train/validation/test partitions before reduction or augmentation. Train generative models only on training data and keep synthetic images out of held-out evaluation.

## LoRA boundary
`lora.py` validates adapter metadata and computes deficits. It does not train or generate images. A faithful training script needs the exact base model/revision, adapter rank/alpha, target modules, optimizer and generation settings. Unknown values are intentionally not invented.

## Verification
Seven reference tests check numerical edge cases, confusion-matrix orientation, all 21 experiment conditions, class weights, missing adapter metadata, real image preprocessing and a dummy-model inference contract. No dataset training, GPU execution or paper-performance reproduction has been performed.

## API references
- https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
