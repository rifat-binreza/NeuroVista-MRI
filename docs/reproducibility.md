# Evidence and reproducibility

| Evidence | Available? | Interpretation |
|---|---|---|
| Published paper metadata and abstract | Yes | Source for the cited results |
| Two inference / visualization notebooks | Privately supplied | Basis for the interface integration |
| Complete training and LoRA pipeline | No | End-to-end reproduction is not possible from this release |
| Verified deployment checkpoints | Not yet | Real inference has not been validated |
| Held-out external evaluation | Not supplied | No external clinical validation claim |

The notebook collection includes randomly generated illustrative Dice scores and random-mask IoU calculations. These are visualization examples, **not empirical model evaluation**, and are excluded from all performance claims here.

The private engineering test suite checks image decoding, input limits, output shape and finite values, unavailable-backend behavior and access-key enforcement using dummy models. Passing these tests does not measure accuracy or reproduce the reported results.

Before real inference is enabled: verify checkpoint provenance and hashes; confirm model shapes, class order, normalization and resize conventions; validate with a known labelled sample set; document package compatibility; test end-to-end request handling on the actual hosting environment.
