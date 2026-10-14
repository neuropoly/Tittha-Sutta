#!/usr/bin/env python3
"""Compute image metrics (SNR, CNR, ...) from a directory of NIfTIs and their segmentations.

Add a metric by writing a function decorated with @metric. It receives a
Context (image data + binary masks) and returns a float. The function name is
used as the column name in the CSV output.
"""
import argparse, csv, sys
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import nibabel as nib
import numpy as np

class Mask(Enum):
    spinalcord = 1 << 0
    graymatter = 1 << 1

class ROI(Enum):
    spinalcord = "spinalcord"
    graymatter = "graymatter"
    whitematter = "whitematter"

MASK_SUFFIXES = {
    Mask.spinalcord: "spinalcord",
    Mask.graymatter: "graymatter",
}

METRICS = {}

def metric(per_slice: bool):
    """Register a metric under its function name.

    per_slice: if True, the metric is emitted once per slice of the volume
    instead of once for the whole volume.
    """
    def decorator(fn: callable):
        METRICS[fn.__name__] = Metric(fn, per_slice)
        return fn
    return decorator

@dataclass
class Metric:
    fn: callable
    per_slice: bool

@dataclass
class Context:
    filename: str      # relative path to .nii file
    data: np.ndarray   # image data
    mask: np.ndarray   # boolean arrays, same shape as data
    segmentations: int # Unsigned integer representing the available segmentations

    def _get_seg_mask(self, mask: Mask) -> np.ndarray:
        if not self.segmentations & mask.value:
            raise ValueError(f"Segmentation {Mask(mask).name} not available for {self.filename}")
        return self.mask & mask.value != 0

    def __getitem__(self, name: ROI) -> np.ndarray:
        """For accessing masked data by attribute"""
        spinalcord = self._get_seg_mask(Mask.spinalcord)
        graymatter = self._get_seg_mask(Mask.graymatter)
        match name:
            case ROI.spinalcord:
                return self.data[spinalcord]
            case ROI.graymatter:
                return self.data[graymatter]
            case ROI.whitematter:
                return self.data[spinalcord & ~graymatter]
            case _:
                raise ValueError(f"Unknown ROI {name}")

    def slice_(self, idx: int):
        """Select slice from the volume and return new Context"""
        return Context(f"{self.filename}",
                       self.data[..., idx], self.mask[..., idx], self.segmentations)

    @property
    def n_slices(self) -> int:
        return self.data.shape[2]

def snr(ctx: Context, roi: ROI):
    data = ctx[roi]

    return data.mean() / data.std()

@metric(per_slice=True)
def SNR_SC(ctx: Context):
    return snr(ctx, ROI.spinalcord)

@metric(per_slice=True)
def SNR_WM(ctx: Context):
    return snr(ctx, ROI.whitematter)

@metric(per_slice=True)
def SNR_GM(ctx: Context):
    return snr(ctx, ROI.graymatter)

def cnr(ctx: Context, roi_a: ROI, roi_b: ROI):
    data_a = ctx[roi_a]
    data_b = ctx[roi_b]

    return np.abs(data_a.mean() - data_b.mean()) / np.sqrt(data_a.std()**2 + data_b.std()**2) / 2**0.5

@metric(per_slice=True)
def CNR_WM_GM(ctx: Context):
    return cnr(ctx, ROI.whitematter, ROI.graymatter)

def entropy(ctx: Context):
    b_max = np.sqrt(np.sum(ctx.data**2))

    non_zero_data = ctx.data[ctx.data != 0]

    e = -np.sum(non_zero_data/b_max * np.log(non_zero_data/b_max))

    return e

@metric(per_slice=True)
def ENTROPY(ctx: Context):
    return entropy(ctx)

def find_images(nifti_dir: Path) -> list[Path]:
    """All NIfTIs under nifti_dir, excluding segmentation files."""
    suffixes = tuple(f"_{s}" for s in MASK_SUFFIXES.values())
    return sorted(
        p for p in nifti_dir.rglob("*")
        if p.name.endswith((".nii", ".nii.gz"))
        and not _stem(p).endswith(suffixes)
    )

def _stem(image: Path) -> str:
    """Filename stem with a trailing .nii/.nii.gz removed."""
    name = image.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name

def load_context(image: Path) -> Context:
    data = nib.load(image).get_fdata()
    mask = np.zeros_like(data, dtype=np.uint64)
    segmentations = 0
    for mask_enum, suffix in MASK_SUFFIXES.items():
        seg = image.parent / f"{_stem(image)}_{suffix}.nii.gz"
        if not seg.exists():
            continue
        mask_data = nib.load(seg).get_fdata() > 0.5
        if mask_data.shape != mask.shape:
            raise ValueError(f"{seg.name}: shape {mask_data.shape} != image shape {mask.shape}")
        mask |= mask_data.astype(np.uint64) * mask_enum.value
        segmentations |= mask_enum.value
    return Context(image, data, mask, segmentations)

def compute_metrics(image: Path) -> dict[str, float]:
    ctx = load_context(image)
    return {name: metric.fn(ctx) for name, metric in METRICS.items()}

def write_csv(nifti_dir: Path, out: Path | None = None) -> None:
    """Write metrics for every image under nifti_dir as CSV (default: stdout).

    Columns are the image filename and the slice index, followed by the metric
    columns. Each case emits one whole-volume row (``slice`` empty; per-slice
    metric cells empty) followed by one row per slice (volume-wise metric cells
    empty).
    """
    f = open(out, "w", newline="") if out else sys.stdout
    w = csv.writer(f)
    w.writerow(["image", "slice", *METRICS])
    for image in find_images(nifti_dir):
        ctx = load_context(image)
        filename_relative = Path(ctx.filename).relative_to(nifti_dir)
        whole = {name: metric.fn(ctx)
                 for name, metric in METRICS.items() if not metric.per_slice}
        w.writerow([filename_relative, "", *(whole.get(k, "") for k in METRICS)])
        for i in range(ctx.n_slices):
            sl = ctx.slice_(i)
            row = {name: metric.fn(sl)
                   for name, metric in METRICS.items() if metric.per_slice}
            w.writerow([filename_relative, i, *(row.get(k, "") for k in METRICS)])
    if out:
        f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nifti_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="CSV output (default: stdout)")
    args = ap.parse_args()

    write_csv(args.nifti_dir, args.out)

if __name__ == "__main__":
    main()