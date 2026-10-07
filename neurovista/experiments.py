"""Describe the thesis's 21 classification conditions; do not fabricate results."""
from dataclasses import dataclass, asdict
import json

LABELS = ("Glioma", "Meningioma", "Pituitary", "No Tumor")

@dataclass(frozen=True)
class Experiment:
    name: str
    method: str
    reduced_class: str | None
    real_counts: dict
    synthetic_counts: dict


def experiment_plan():
    """Return original + 4 reduced + 16 intervention conditions.

    Real counts record unique retained images, not oversampled exposures.
    Random augmentation/weighting/oversampling add no synthetic-source images.
    Patient splits must be fixed BEFORE reduction or generation.
    """
    yield Experiment("original", "original", None, dict.fromkeys(LABELS,500), dict.fromkeys(LABELS,0))
    for label in LABELS:
        for method in ("reduced", "lora", "random_augmentation", "class_weighting", "oversampling"):
            counts = dict.fromkeys(LABELS,500); counts[label] = 250
            generated = dict.fromkeys(LABELS,0)
            if method == "lora": generated[label] = 250
            yield Experiment(f"{label.lower().replace(' ', '_')}_{method}", method, label, counts, generated)


def balanced_class_weights(counts):
    """Standard N/(K*n_k) weights; an explicit reconstruction convention."""
    if not counts or any(isinstance(n,bool) or not isinstance(n,int) or n <= 0 for n in counts.values()):
        raise ValueError("Counts must be positive integers")
    total = sum(counts.values())
    return {label: total/(len(counts)*count) for label,count in counts.items()}

if __name__ == "__main__":
    print(json.dumps([asdict(row) for row in experiment_plan()], indent=2))
