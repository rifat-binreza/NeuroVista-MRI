# Inference implementation notes

Technical provenance and validation details for developers. For the project overview, see [NeuroVista MRI](../README.md).

## Input conversion and reference behavior

The documented [public compatibility adapter](neurovista/hf_compatibility.py) and [regression tests](tests/test_hf_compatibility.py) are included. Use `prepare_reference_input(path)` with your trusted model, or `classify_reference(path, model)` for the reference checkpoint’s verified output order. The existing generic `inference.py` remains available for other caller-verified models.

The shared input decoder matches Gradio 5.37.0’s `Image(type="filepath", image_mode="RGB")` behavior. RGB files pass through; other modes are converted to RGB. Grayscale JPEG conversion triggers a second JPEG encode in the original app, so the compatibility path reproduces that step in memory. PNG conversion is lossless. This applies to all supported uploads, with no filename-specific rules or score adjustments.

Independent comparisons against the actual Gradio component produced identical model-input tensors across eight cases covering grayscale/RGB JPEG, PNG, alpha and EXIF orientation. The supplied comparison JPEG produced 99.7326% for the No Tumor class when decoded directly and 99.8394% after the reference conversion, explaining the displayed 99.73% versus 99.84% difference. Both checkpoint hashes match the original. Matching software behavior does not establish clinical correctness.

## Input screening and uncertain results

The demo rejects obvious color images, blank/low-contrast images, tiny inputs and extreme aspect ratios before running the models. You must confirm that the input is an original, de-identified brain MRI slice. Rejected inputs show no class scores or heatmaps.

Ambiguous classifier outputs are withheld when the top score is below 0.70 or the top-two margin is below 0.20. These are **uncalibrated display-policy thresholds**, not validated clinical confidence or an out-of-distribution detector. High scores do not prove that an image is an MRI. Grayscale photos, CT images and diagrams may pass the basic checks; do not convert photos to grayscale to bypass them.

Deployment regression checks confirmed that the supplied portrait example is rejected (HTTP 422, no predictions) and an MRI example from the supplied notebook reaches inference. Eight policy/API tests passed. These limited checks do not establish diagnostic accuracy or MRI-detection sensitivity/specificity.


## Output interpretation

The segmentation checkpoint produces three sigmoid channels. The displayed JET heatmap preserves the supplied interface's visualization; it is not a validated binary tumor boundary or channel-to-class mapping. The classification output indices are Glioma, Meningioma, No Tumor, Pituitary. The source interface is linked in the adapter documentation; full publication attribution is preserved in CITATION.cff and citation.bib.
