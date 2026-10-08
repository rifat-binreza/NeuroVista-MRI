<p align="center"><img src="assets/banner.svg" width="100%" alt="NeuroVista MRI — Rifat Bin Reza — Segmentation, Classification, Augmentation" /></p>
<p align="center"><a href="https://doi.org/10.1109/IICAIET67254.2025.11264978"><img src="https://img.shields.io/badge/IEEE-IICAIET_2025-00b9a8?style=for-the-badge" alt="IEEE IICAIET 2025"></a> <img src="https://img.shields.io/badge/RELEASE-RESEARCH_SHOWCASE-5964ec?style=for-the-badge" alt="Research showcase"></p>

<h1 align="center">NeuroVista MRI</h1>
<p align="center"><b>Localize the tumor. Classify the image. Study the imbalance.</b><br>A personal research showcase by <a href="https://github.com/rifat-binreza">Rifat Bin Reza</a>, co-author of published IEEE IICAIET 2025 research.</p>

<p align="center"><a href="#published-evidence">Results</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/reproducibility.md">Reproducibility</a> · <a href="#publication--credit">Publication</a> · <a href="docs/access-and-rights.md">Access & rights</a></p>

## The research

Brain MRI analysis brings together two questions: **where is the predicted tumor region, and which class does the image resemble?** Our published work studies U-Net segmentation and Xception classification, with LoRA-driven synthetic augmentation to address class imbalance.

NeuroVista MRI is my personal research portfolio for brain MRI analysis. The accompanying interface is designed around an MRI workspace, segmentation overlays, four-class scores and a publication explorer.

> **Live research demo:** [Open NeuroVista MRI](https://neuro-vista-mri-private.vercel.app) — deployed on Vercel under Rifat1 with the supplied trained checkpoints. Upload a de-identified PNG/JPEG under 2 MB, confirm permission, then select **Run research analysis**. No separate demo key is required; sign in to Vercel if prompted. First use may take longer while models load.

## Explore the Python implementation

These are **new reference implementations reconstructed from the supplied notebooks and thesis**, not recovered original training code. They make the methods inspectable without exposing the complete application, model weights or private download links.

| Module | What it implements |
|---|---|
| [`preprocessing.py`](neurovista/preprocessing.py) | RGB conversion, resizing, normalization and binary-mask preparation |
| [`unet.py`](neurovista/unet.py) | Configurable reference U-Net and soft Dice loss |
| [`xception.py`](neurovista/xception.py) | Xception backbone and four-class classification head |
| [`metrics.py`](neurovista/metrics.py) | Dice, IoU, confusion matrix, precision, recall and F1 |
| [`experiments.py`](neurovista/experiments.py) | All 21 thesis classification conditions and class weighting |
| [`lora.py`](neurovista/lora.py) | Adapter provenance checks, file hashing and augmentation deficit planning |
| [`inference.py`](neurovista/inference.py) | Local inference with caller-supplied models and verified class order |

```sh
python -m pip install -r requirements.txt
python -m examples.inspect_plan
python -m unittest discover -s tests -v
```

The plan and tests run without TensorFlow or model downloads. Model construction requires TensorFlow. Read the [code guide](docs/code-guide.md) and [source-to-code decisions](docs/source-to-code.md) before using a checkpoint. LoRA training is not falsely represented as reconstructed: essential adapter settings are absent.

## Published evidence

| Experiment | Paper-reported result |
| :--- | ---: |
| U-Net segmentation — Dice | **88.43%** |
| U-Net segmentation — IoU | **84.21%** |
| Glioma classification — original dataset | **96.23%** |
| Glioma classification — reduced dataset | **92.88%** |
| Glioma classification — LoRA-augmented dataset | **95.53%** |

The abstract reports average improvements over reduced datasets of **2.55% accuracy, 2.38% precision, 2.37% recall and 2.51% F1-score**. Values preserve the abstract’s wording; they are not newly reproduced benchmarks. See [machine-readable results](docs/paper-results.csv).

## How the supplied demo operates

```mermaid
flowchart TD
    A["MRI image"] --> B["Resize: 256 × 256"]
    A --> C["Resize: 299 × 299"]
    B --> D["U-Net → three-channel heatmap"]
    C --> E["Xception → four class scores"]
```

The supplied inference code runs the two branches independently. It does **not** pass the segmentation mask into Xception. LoRA augmentation belongs to the paper’s training experiments; it is not a live generation feature in this interface.

The deployed checkpoints were loaded and tested through the live API with a synthetic fixture on 8 October 2026. The request returned four finite class scores and a PNG heatmap; this checks execution, not diagnostic accuracy. Segmentation produces three sigmoid channels. The app preserves the supplied notebook’s OpenCV JET visualization rather than claiming a validated binary boundary or assigning tumor labels to those channels. Class order follows the supplied notebook; independent training-label verification remains outstanding.

## A deliberate public release

This repository contains documented Python reference modules, the research story, original vector artwork, citation metadata and reported results. Training notebooks, application internals, model weights and direct model-download links remain outside this public repository.

| Included here | Kept private |
| :--- | :--- |
| Documented Python reference modules | Original notebooks |
| Methodology and reported metrics | Model checkpoints and download pointers |
| Reproducibility and evidence notes | Backend and deployment implementation |
| Citation and access policy | Secrets and access keys |

The implementation package adds bounded image validation, a private Python inference endpoint, checkpoint integrity checks, request isolation and an explicit unavailable state. These engineering checks do not constitute clinical validation or reproduction of the paper.

## Publication & credit

**Deep Learning for Brain Tumor Detection: U-Net Segmentation and Xception Classification with LoRA-Driven Synthetic Data Augmentation**  
2025 IEEE International Conference on Artificial Intelligence in Engineering and Technology (**IICAIET**)

**Rifat Bin Reza** · Electrical and Electronic Engineering, BRAC University  
This repository presents my research portfolio and a documented reconstruction of the MRI analysis workflow. The complete publication attribution is preserved in the citation files.

**DOI:** [10.1109/IICAIET67254.2025.11264978](https://doi.org/10.1109/IICAIET67254.2025.11264978)

Use [CITATION.cff](CITATION.cff) or [BibTeX](citation.bib) to cite the paper. For research inquiries, reach me through [my GitHub profile](https://github.com/rifat-binreza) or explore [my Google Scholar profile](https://scholar.google.com/citations?user=U7HsBd4AAAAJ&hl=en).

## Responsible interpretation

This is a research demonstration, not a diagnostic system. Model scores are not calibrated probabilities of disease. Clinical use, external generalization and regulatory readiness are not established by this release. Do not submit identifiable patient information.

## Rights

No open-source license is granted for the unpublished implementation or models. See [RIGHTS.md](RIGHTS.md). The IEEE publication and third-party materials retain their respective rights; the full paper is not redistributed here.
