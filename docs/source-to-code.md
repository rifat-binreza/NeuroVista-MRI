# Source-to-code decisions

The user supplied two inference/visualization notebooks and a thesis draft supporting the conference research. The thesis is a separately authored degree submission; it is a technical reference, not evidence of sole authorship by this repository's maintainer. The full thesis and paper are not republished here.

| Source | Observed detail | Implementation decision |
|---|---|---|
| Thesis §3.3 / Table 4.6 | RGB 256×256, /255, Adam 1e-4, batch 16, 50 epochs, Dice loss | Documented; architecture supplied as a new reference builder |
| Thesis §3.3.2 | Encoder filters 64–512, three output channels, ambiguous parameter accounting | Require explicit channels/activation; do not claim exact architecture or weight compatibility |
| Supplied inference notebooks | Parallel resized image branches; mask not passed to classifier | Preserve independent inference semantics |
| Thesis §3.6 / Table 4.8 | Xception ImageNet, flatten, dropout .5, dense128, softmax4; Adam1e-4, batch32, 50 epochs | Reference builder; exact dropout order and freezing schedule marked as new choices |
| Thesis §3.4 / Table 4.7 | 100 LoRA training images/class, 12 epochs, 10 repeats, learning rate1e-5 | Preserve reported settings; require missing rank/alpha/base revision rather than guessing |
| Thesis §3.6.3 | One original, four reduced and sixteen intervention models | Executable 21-condition experiment plan |
| Notebook visualization examples | Random Dice values and random masks | Excluded from research results |

## Unresolved segmentation schema
The thesis describes three final channels. The private web adapter currently accepts only a single-channel probability mask and will reject a three-channel checkpoint. This deliberate restriction must be resolved against the actual model and label schema before activation; selecting or averaging channels without evidence is not acceptable.

## Provenance categories
- **Notebook-grounded:** resize sizes, RGB, /255, independent branches.
- **Thesis-grounded:** architecture families, reported hyperparameters, imbalance experiment counts.
- **New reconstruction:** precise block layout, skip concatenation, dropout placement, metric zero-division rules, explicit validation and helper interfaces.
- **Unavailable:** original training scripts, verified splits, complete LoRA configuration, trained checkpoint verification and attributable individual contribution breakdown.

No new module is represented as historical code used to obtain the published numbers. No specific research subtask is assigned solely to an individual without evidence.
