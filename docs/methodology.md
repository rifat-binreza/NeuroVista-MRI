# Methodology and implementation boundary

## Publication-level method
The paper studies U-Net segmentation, Xception classification and LoRA-driven synthetic MRI augmentation. To study imbalance, a class is reduced at a time and classification is compared before and after augmentation. The abstract reports aggregate improvements and the glioma example reproduced in this repository.

## Supplied inference notebooks
| Branch | Image input | Observed preprocessing | Output |
|---|---|---|---|
| U-Net | RGB, 256 × 256 | Divide pixel values by 255 | Spatial prediction |
| Xception | RGB, 299 × 299 | Divide pixel values by 255 | Four class scores |

Observed class-label order: Glioma, Meningioma, Pituitary, No Tumor. This order and preprocessing must be checked against the trained artifacts before enabling a new deployment. No assumption is made that default Xception preprocessing can replace the supplied normalization.

The branches take resized versions of the same image. The provided code does not use the predicted mask as classifier input. The new interface uses a threshold of 0.5 for its display mask; this is an interface convention, not a claim about the paper’s evaluation protocol.

## LoRA scope
The available notebooks do not contain the complete generative-model fine-tuning or classifier training pipeline. Accordingly, this release does not claim to reproduce LoRA training, dataset partitions, model selection or the paper’s evaluation.
