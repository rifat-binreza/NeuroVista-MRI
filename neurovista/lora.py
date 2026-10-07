"""LoRA artifact validation and generation planning, not invented training code.

The thesis reports Tensor.Art, 100 images/class, 12 epochs, 10 repeats and
learning rate 1e-5. Rank, alpha, target modules and exact base revision are
missing. It calls the generation model 'Flux 0.1 Dev'; no model ID is inferred.
"""
from pathlib import Path
import hashlib

REPORTED_SETTINGS = {"images_per_class":100, "epochs":12, "repeats":10, "learning_rate":1e-5}


def validate_adapter_manifest(manifest):
    """Require caller-supplied provenance before integrating an adapter.

    This validates completeness only; it cannot certify model compatibility.
    """
    keys = ("base_model", "base_revision", "rank", "alpha", "target_modules", "adapter_sha256")
    missing = [key for key in keys if not manifest.get(key)]
    if missing: raise ValueError("Missing verified LoRA details: " + ", ".join(missing))
    if type(manifest["rank"]) is not int or manifest["rank"] <= 0:
        raise ValueError("rank must be a positive integer")
    digest = manifest["adapter_sha256"]
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError("Use a lowercase SHA-256 digest")
    return dict(manifest)


def sha256_file(path):
    """Fingerprint a local adapter without uploading it or exposing a URL."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def generation_deficit(real_count, target_count=500):
    """Plan how many reviewed synthetic training images are needed."""
    if type(real_count) is not int or type(target_count) is not int or min(real_count,target_count)<0:
        raise ValueError("Counts must be nonnegative integers")
    return max(0, target_count-real_count)
