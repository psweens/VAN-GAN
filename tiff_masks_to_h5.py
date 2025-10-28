"""Utility to convert a directory of TIFF segmentation masks into HDF5 volumes.

This script is tailored for VAN-GAN's synthetic data pipeline, which expects
segmentation datasets stored under the "label" key with a trailing channel
axis.  It scans an input directory for ``.tif``/``.tiff`` files, converts each
mask to ``float32`` (optionally normalising the values), adds the required
channel dimension, and writes the result to an ``.h5`` file in the output
folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import tifffile

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalise_mask(mask: np.ndarray) -> np.ndarray:
    """Normalise mask values to the ``[0, 1]`` range if they are not already.

    ``tifffile`` preserves the on-disk dtype, which for binary masks is often
    ``uint8`` with values in ``{0, 255}``.  The training pipeline expects
    floating-point masks; therefore we convert to ``float32`` and, if
    necessary, scale the values so the maximum becomes ``1``.
    """

    mask = mask.astype(np.float32, copy=False)
    max_val = np.nanmax(mask)
    if max_val > 1.0:
        mask /= max_val
    return mask


def _ensure_channel_axis(mask: np.ndarray, move_first_axis: bool) -> np.ndarray:
    """Ensure that ``mask`` has a trailing singleton channel axis.

    Parameters
    ----------
    mask:
        The array returned by :func:`tifffile.imread`.
    move_first_axis:
        If ``True`` and the mask is 3-D, move the first axis (commonly the slice
        axis in ``Z, Y, X`` TIFF stacks) to the end before appending the channel
        dimension.  This matches the ``(X, Y, Z, C)`` layout used by VAN-GAN's
        loaders.
    """

    if mask.ndim == 2:
        return mask[..., np.newaxis]

    if mask.ndim == 3:
        # Heuristic: if the last axis looks like a colour/channel dimension,
        # preserve it; otherwise treat the array as a stack of monochrome
        # slices and add an explicit channel.
        if mask.shape[-1] <= 4:  # e.g. RGB, RGBA or already single channel
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                raise ValueError(
                    "Multi-channel masks are not supported. Found shape "
                    f"{mask.shape}."
                )
            return mask[..., np.newaxis]

        if move_first_axis:
            mask = np.moveaxis(mask, 0, -1)
        return mask[..., np.newaxis]

    raise ValueError(
        "Masks must be 2-D or 3-D arrays. Received array with "
        f"{mask.ndim} dimensions."
    )


def _suggest_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Derive chunk sizes similar to the preprocessing pipeline.

    The heuristic mirrors :meth:`Preprocessor._auto_chunks` by constraining each
    chunk dimension to be at most 128 voxels while staying a power of two.
    """

    chunks = []
    for dim in shape:
        chunk = 1
        while chunk * 2 <= dim and chunk < 128:
            chunk *= 2
        chunks.append(chunk)
    return tuple(chunks)


# -----------------------------------------------------------------------------
# Conversion routine
# -----------------------------------------------------------------------------

def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str,
    move_first_axis: bool,
    normalise: bool,
    compression: str,
) -> None:
    """Convert all TIFF masks in ``input_dir`` into HDF5 files."""

    tiff_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".tif", ".tiff"}
    )

    if not tiff_paths:
        raise FileNotFoundError(
            f"No TIFF files found in {input_dir!s}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for tiff_path in tiff_paths:
        mask = tifffile.imread(tiff_path)
        if normalise:
            mask = _normalise_mask(mask)
        else:
            mask = mask.astype(np.float32, copy=False)

        mask = _ensure_channel_axis(mask, move_first_axis)
        chunks = _suggest_chunks(mask.shape)

        h5_path = output_dir / f"{tiff_path.stem}.h5"
        with h5py.File(h5_path, "w") as h5f:
            h5f.create_dataset(
                dataset_name,
                data=mask,
                chunks=chunks,
                compression=compression,
                shuffle=True,
            )
        print(f"Converted {tiff_path.name} -> {h5_path.name} (shape={mask.shape}).")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a folder of TIFF masks into VAN-GAN compatible HDF5 files.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .tif/.tiff masks.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory for the generated .h5 files.",
    )
    parser.add_argument(
        "--dataset-name",
        default="label",
        help="Name of the dataset stored inside each HDF5 file (default: %(default)s).",
    )
    parser.add_argument(
        "--no-normalise",
        dest="normalise",
        action="store_false",
        help="Disable normalisation to the [0, 1] range.",
    )
    parser.add_argument(
        "--keep-first-axis",
        dest="move_first_axis",
        action="store_false",
        help=(
            "Do not move the first axis of 3-D TIFF stacks to the end before adding the"
            " channel dimension."
        ),
    )
    parser.add_argument(
        "--compression",
        default="lzf",
        help="Compression algorithm used by h5py (default: %(default)s).",
    )
    parser.set_defaults(normalise=True, move_first_axis=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_directory(
        args.input_dir,
        args.output_dir,
        dataset_name=args.dataset_name,
        move_first_axis=args.move_first_axis,
        normalise=args.normalise,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
