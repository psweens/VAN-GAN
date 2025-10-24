"""Train and evaluate a paired 3-D U-Net baseline.

This script mirrors the data ingestion and sliding-window inference
pipeline used by VAN-GAN so that a supervised 3-D U-Net can be trained and
evaluated on the same paired datasets.  It expects HDF5 volumes with
`image` and `label` datasets (identical to the GAN pipeline) stored in
separate folders for the imaging and segmentation domains.  The script can
either ingest explicit ``train``/``val``/``test`` sub-directories or, more
commonly, consume flat directories of HDF5 volumes and automatically split
them into train/validation/test partitions (ensuring paired filenames
between the imaging and segmentation folders).  After training, the best
validation checkpoint is reused for sliding-window inference across every
split and over the original bioimage directories.

Example
-------
python unet_baseline.py \
    --image-dir ./data/bioimage \
    --label-dir ./data/segmentation \
    --output-dir ./unet_baseline_output
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf

from cbDice_func import soft_dice_cbdice_loss
from utils import min_max_norm_tf


# ---------------------------------------------------------------------------
#                            Sliding window helpers
# ---------------------------------------------------------------------------


def _gaussian_window(shape: Sequence[int], sigma: float = 1.0) -> np.ndarray:
    if len(shape) != 3:
        raise ValueError("Gaussian window expects a 3-D shape")
    axes = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*axes, indexing="ij")
    kernel = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    kernel /= kernel.max()
    return kernel[..., np.newaxis]


def _hann_window(shape: Sequence[int]) -> np.ndarray:
    if len(shape) != 3:
        raise ValueError("Hann window expects a 3-D shape")
    windows = [np.hanning(s) for s in shape]
    kernel = np.outer(windows[0], windows[1]).reshape(shape[0], shape[1], 1)
    kernel *= windows[2][np.newaxis, np.newaxis, :]
    kernel /= kernel.max()
    return kernel[..., np.newaxis]


def _tukey_window(shape: Sequence[int], alpha: float = 0.5) -> np.ndarray:
    if len(shape) != 3:
        raise ValueError("Tukey window expects a 3-D shape")
    from scipy.signal import tukey

    windows = [tukey(s, alpha=alpha) for s in shape]
    kernel = np.outer(windows[0], windows[1]).reshape(shape[0], shape[1], 1)
    kernel *= windows[2][np.newaxis, np.newaxis, :]
    kernel /= kernel.max()
    return kernel[..., np.newaxis]


def _get_window(window_type: str, shape: Sequence[int], **kwargs) -> np.ndarray:
    window_type = window_type.lower()
    if window_type == "gaussian":
        return _gaussian_window(shape, sigma=kwargs.get("sigma", 1.0))
    if window_type == "hann":
        return _hann_window(shape)
    if window_type == "tukey":
        return _tukey_window(shape, alpha=kwargs.get("alpha", 0.5))
    raise ValueError(f"Unsupported window type: {window_type}")


def sliding_window_inference(
    volume: np.ndarray,
    model: tf.keras.Model,
    patch_shape: Sequence[int],
    stride: Sequence[int],
    *,
    batch_size: int = 4,
    window_type: str = "tukey",
    window_kwargs: Optional[dict] = None,
    preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Replicate VAN-GAN's sliding-window + apodisation inference pipeline."""

    if window_kwargs is None:
        window_kwargs = {}

    if volume.ndim != 4:
        raise ValueError("Expected volume to have shape (H, W, D, C)")

    orig_H, orig_W, orig_D, _ = volume.shape

    pad = [max(0, patch_shape[i] - volume.shape[i]) for i in range(3)]
    if any(pad):
        padding = [(0, pad[0]), (0, pad[1]), (0, pad[2]), (0, 0)]
        volume = np.pad(volume, padding, mode="reflect")
        H, W, D, _ = volume.shape
    else:
        H, W, D = orig_H, orig_W, orig_D

    pred = np.zeros_like(volume, dtype=np.float32)
    weights = np.zeros_like(volume, dtype=np.float32)

    win_shape = (
        patch_shape[0],
        patch_shape[1],
        patch_shape[2] if patch_shape[2] != D else 1,
    )
    window_np = _get_window(window_type, win_shape, **window_kwargs).astype(np.float32)
    window_tf = tf.convert_to_tensor(window_np, dtype=tf.float32)

    if patch_shape[2] == D:
        window_np = np.repeat(window_np[:, :, np.newaxis, :], D, axis=2)
        window_tf = tf.repeat(window_tf[:, :, tf.newaxis, :], D, axis=2)

    row_steps = list(range(0, H - patch_shape[0] + 1, stride[0]))
    col_steps = list(range(0, W - patch_shape[1] + 1, stride[1]))
    if patch_shape[2] == D:
        dep_steps = [0]
    else:
        dep_steps = list(range(0, D - patch_shape[2] + 1, stride[2]))

    batch_subvols: List[np.ndarray] = []
    batch_meta: List[Tuple[slice, slice, slice, Tuple[int, int, int]]] = []

    def _flush_batch():
        nonlocal batch_subvols, batch_meta, pred, weights
        if not batch_subvols:
            return

        batch_arr = np.stack(batch_subvols, axis=0).astype(np.float32)
        if preprocess is not None:
            batch_arr = preprocess(batch_arr)

        preds_tf = model(tf.convert_to_tensor(batch_arr), training=False)
        preds_tf *= window_tf
        preds_np = preds_tf.numpy()

        for patch_pred, (rs, cs, ds, orig_shape) in zip(preds_np, batch_meta):
            hr, wr, dr = orig_shape
            pred[rs, cs, ds] += patch_pred[:hr, :wr, :dr]
            weights[rs, cs, ds] += window_np[:hr, :wr, :dr]

        batch_subvols.clear()
        batch_meta.clear()

    for r in row_steps:
        for c in col_steps:
            for d in dep_steps:
                rs = slice(r, min(r + patch_shape[0], H))
                cs = slice(c, min(c + patch_shape[1], W))
                ds = slice(d, min(d + patch_shape[2], D))

                subvol = volume[rs, cs, ds]

                pad_h = patch_shape[0] - subvol.shape[0]
                pad_w = patch_shape[1] - subvol.shape[1]
                pad_d = patch_shape[2] - subvol.shape[2]
                if pad_h or pad_w or pad_d:
                    pad_dims = [(0, pad_h), (0, pad_w), (0, pad_d), (0, 0)]
                    subvol = np.pad(subvol, pad_dims, mode="reflect")

                batch_subvols.append(subvol)
                batch_meta.append((rs, cs, ds, (
                    rs.stop - rs.start,
                    cs.stop - cs.start,
                    ds.stop - ds.start,
                )))

                if len(batch_subvols) == batch_size:
                    _flush_batch()

    _flush_batch()
    np.divide(pred, np.maximum(weights, 1e-6), out=pred)

    if any(pad):
        return pred[:orig_H, :orig_W, :orig_D, :]
    return pred


# ---------------------------------------------------------------------------
#                         Dataset preparation utilities
# ---------------------------------------------------------------------------


def _list_h5_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(directory)
    files = sorted([p for p in directory.iterdir() if p.suffix in {".h5", ".hdf5"}])
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {directory}")
    return files


def _load_random_paired_patch(
    image_path: bytes,
    label_path: bytes,
    patch_shape: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    image_path = image_path.decode()
    label_path = label_path.decode()

    with h5py.File(image_path, "r") as f_img, h5py.File(label_path, "r") as f_lbl:
        img_ds = f_img["image"]
        lbl_ds = f_lbl["label"]

        if img_ds.shape[:3] != lbl_ds.shape[:3]:
            raise ValueError("Image and label volumes must share spatial dimensions")

        max_offsets = [max(0, s - p) for s, p in zip(img_ds.shape[:3], patch_shape)]
        offsets = [np.random.randint(0, m + 1) if m > 0 else 0 for m in max_offsets]

        slices = []
        for off, size, dim in zip(offsets, patch_shape, img_ds.shape[:3]):
            end = min(off + size, dim)
            slices.append(slice(off, end))
        slices = tuple(slices)

        img_patch = img_ds[slices + (slice(None),)]
        lbl_patch = lbl_ds[slices + (slice(None),)]

        pad_dims = []
        need_pad = False
        for size, actual in zip(patch_shape, img_patch.shape[:3]):
            pad_total = size - actual
            if pad_total > 0:
                need_pad = True
            pad_dims.append((0, max(0, pad_total)))
        pad_dims.append((0, 0))
        if need_pad:
            img_patch = np.pad(img_patch, pad_dims, mode="reflect")
            lbl_patch = np.pad(lbl_patch, pad_dims, mode="reflect")

    return img_patch.astype(np.float32), lbl_patch.astype(np.float32)


def _load_full_volume(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        data = f[dataset][...].astype(np.float32)
    return data


@dataclass
class PairedDataset:
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    steps_per_epoch: int
    validation_steps: int


@dataclass
class SplitPaths:
    images: List[Path]
    labels: List[Path]


def _build_tf_dataset(
    image_paths: Sequence[Path],
    label_paths: Sequence[Path],
    patch_shape: Sequence[int],
    batch_size: int,
) -> tf.data.Dataset:
    paths_dataset = tf.data.Dataset.from_tensor_slices(
        (list(map(str, image_paths)), list(map(str, label_paths)))
    )

    def _loader(img_path, lbl_path):
        img, lbl = tf.numpy_function(
            _load_random_paired_patch,
            [img_path, lbl_path, patch_shape],
            (tf.float32, tf.float32),
        )
        img.set_shape(patch_shape + (1,))
        lbl.set_shape(patch_shape + (1,))
        img = min_max_norm_tf(img, axis=None)
        return img, lbl

    dataset = (
        paths_dataset.shuffle(len(image_paths))
        .repeat()
        .map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


def _validate_pairs(image_paths: Sequence[Path], label_paths: Sequence[Path], split: str) -> None:
    if len(image_paths) != len(label_paths):
        raise ValueError(f"{split.capitalize()} image/label counts do not match")
    for img, lbl in zip(image_paths, label_paths):
        if img.stem != lbl.stem:
            raise ValueError(f"Mismatched {split} pair: {img.name} vs {lbl.name}")


def _match_paired_roots(image_root: Path, label_root: Path) -> List[Tuple[Path, Path]]:
    image_files = _list_h5_files(image_root)
    label_files = _list_h5_files(label_root)

    labels_by_stem = {p.stem: p for p in label_files}
    pairs: List[Tuple[Path, Path]] = []
    missing_labels: List[str] = []

    for img_path in image_files:
        lbl_path = labels_by_stem.get(img_path.stem)
        if lbl_path is None:
            missing_labels.append(img_path.name)
        else:
            pairs.append((img_path, lbl_path))

    if missing_labels:
        raise ValueError(
            "Missing matching label files for: " + ", ".join(sorted(missing_labels))
        )

    extra_labels = sorted(set(labels_by_stem) - {img.stem for img, _ in pairs})
    if extra_labels:
        raise ValueError(
            "Label files without corresponding images: " + ", ".join(extra_labels)
        )

    return pairs


def _fractional_split(
    pairs: Sequence[Tuple[Path, Path]],
    split_names: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, SplitPaths]:
    if len(split_names) != 3:
        raise ValueError("Expected exactly three split names (train/val/test)")
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Validation and test fractions must be non-negative")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1")

    total = len(pairs)
    if total == 0:
        raise ValueError("No paired HDF5 volumes found for splitting")

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    val_count = int(np.floor(total * val_fraction))
    test_count = int(np.floor(total * test_fraction))
    train_count = total - val_count - test_count

    # Guarantee at least one sample per split when possible.
    if train_count <= 0:
        train_count = 1
        if val_count > test_count:
            val_count = max(0, val_count - 1)
        else:
            test_count = max(0, test_count - 1)
    if val_fraction > 0 and val_count == 0 and total >= 2:
        val_count = 1
        train_count = max(1, train_count - 1)
    if test_fraction > 0 and test_count == 0 and total - val_count >= 2:
        test_count = 1
        train_count = max(1, train_count - 1)

    if val_fraction > 0 and val_count == 0:
        raise ValueError(
            "Validation split is empty; increase dataset size or adjust --val-fraction"
        )
    if test_fraction > 0 and test_count == 0:
        raise ValueError(
            "Test split is empty; increase dataset size or adjust --test-fraction"
        )

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count : train_count + val_count + test_count]

    split_map: Dict[str, SplitPaths] = {}
    split_indices = {
        split_names[0]: train_idx,
        split_names[1]: val_idx,
        split_names[2]: test_idx,
    }

    for split_name, split_idx in split_indices.items():
        images = [pairs[i][0] for i in split_idx]
        labels = [pairs[i][1] for i in split_idx]
        split_map[split_name] = SplitPaths(images=images, labels=labels)

    return split_map


def discover_split_paths(
    image_root: Path,
    label_root: Path,
    split_names: Sequence[str],
    *,
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> Dict[str, SplitPaths]:
    subdirs_exist = all((image_root / split).exists() and (label_root / split).exists() for split in split_names)
    if subdirs_exist:
        split_map: Dict[str, SplitPaths] = {}
        for split in split_names:
            img_dir = image_root / split
            lbl_dir = label_root / split
            images = _list_h5_files(img_dir)
            labels = _list_h5_files(lbl_dir)
            _validate_pairs(images, labels, split)
            split_map[split] = SplitPaths(images=images, labels=labels)
        return split_map

    pairs = _match_paired_roots(image_root, label_root)
    return _fractional_split(pairs, split_names, val_fraction, test_fraction, split_seed)


def prepare_datasets(
    train_split: SplitPaths,
    val_split: SplitPaths,
    patch_shape: Sequence[int],
    batch_size: int,
) -> PairedDataset:
    train_ds = _build_tf_dataset(train_split.images, train_split.labels, patch_shape, batch_size)
    val_ds = _build_tf_dataset(val_split.images, val_split.labels, patch_shape, batch_size)

    steps_per_epoch = max(1, len(train_split.images) // batch_size)
    validation_steps = max(1, len(val_split.images) // batch_size)

    return PairedDataset(
        train_dataset=train_ds,
        val_dataset=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )


# ---------------------------------------------------------------------------
#                                Model
# ---------------------------------------------------------------------------


def conv_block(x, filters: int, *, kernel_size: int = 3, dropout: float = 0.0) -> tf.Tensor:
    x = tf.keras.layers.Conv3D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv3D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_unet(
    input_shape: Sequence[int],
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.0,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    skips = []
    x = inputs

    for d in range(depth):
        filters = base_filters * (2 ** d)
        x = conv_block(x, filters, dropout=dropout if d > 0 else 0.0)
        skips.append(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)

    filters = base_filters * (2 ** depth)
    x = conv_block(x, filters, dropout=dropout)

    for d in reversed(range(depth)):
        filters = base_filters * (2 ** d)
        x = tf.keras.layers.Conv3DTranspose(filters, 2, strides=2, padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skips[d]])
        x = conv_block(x, filters, dropout=dropout if d > 0 else 0.0)

    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="unet3d")


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, epsilon: float = 1e-5) -> tf.Tensor:
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return tf.reduce_mean(dice)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred)


# ---------------------------------------------------------------------------
#                                Training
# ---------------------------------------------------------------------------


def configure_gpus():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


def train_unet(
    args: argparse.Namespace,
    split_map: Dict[str, SplitPaths],
    patch_shape: Sequence[int],
) -> tf.keras.Model:
    strategy = configure_gpus()

    dataset = prepare_datasets(
        split_map[args.train_split],
        split_map[args.val_split],
        patch_shape,
        args.batch_size,
    )

    with strategy.scope():
        model = build_unet(patch_shape + (1,), base_filters=args.base_filters, depth=args.depth, dropout=args.dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        cb_loss_fn = soft_dice_cbdice_loss()

        def loss_with_boundary(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return cb_loss_fn(y_true, y_pred, dist_thr=20.0)

        loss_with_boundary.__name__ = "soft_dice_cbdice_loss"

        model.compile(optimizer=optimizer, loss=loss_with_boundary, metrics=[dice_coefficient])

    callbacks = []
    best_ckpt_path = Path(args.output_dir) / "unet_best.h5"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_ckpt_path),
        save_best_only=True,
        monitor="val_dice_coefficient",
        mode="max",
    ))

    model.fit(
        dataset.train_dataset,
        epochs=args.epochs,
        steps_per_epoch=dataset.steps_per_epoch,
        validation_data=dataset.val_dataset,
        validation_steps=dataset.validation_steps,
        callbacks=callbacks,
    )

    if best_ckpt_path.exists():
        model.load_weights(str(best_ckpt_path))

    return model


# ---------------------------------------------------------------------------
#                                Evaluation
# ---------------------------------------------------------------------------


def _preprocess_numpy(arr: np.ndarray) -> np.ndarray:
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    tensor = min_max_norm_tf(tensor)
    return tensor.numpy()


def run_inference_on_split(
    model: tf.keras.Model,
    args: argparse.Namespace,
    split_name: str,
    image_paths: Sequence[Path],
    *,
    label_paths: Optional[Sequence[Path]] = None,
) -> List[dict]:
    patch_shape = tuple(args.patch_size)
    stride = tuple(args.inference_stride)

    metrics: List[dict] = []
    window_kwargs = json.loads(args.window_kwargs) if args.window_kwargs else None
    predictions_root = Path(args.output_dir) / "predictions" / split_name
    if args.save_predictions:
        predictions_root.mkdir(parents=True, exist_ok=True)
    else:
        predictions_root.parent.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        volume = _load_full_volume(img_path, "image")
        if volume.ndim == 3:
            volume = volume[..., np.newaxis]

        prediction = sliding_window_inference(
            volume,
            model,
            patch_shape,
            stride,
            batch_size=args.inference_batch_size,
            window_type=args.window_type,
            window_kwargs=window_kwargs,
            preprocess=lambda arr: _preprocess_numpy(arr).astype(np.float32),
        )

        prob = prediction[..., 0]
        pred_mask = (prob > args.threshold).astype(np.float32)

        if args.save_predictions:
            pred_path = predictions_root / f"{img_path.stem}_prediction.h5"
            with h5py.File(pred_path, "w") as f:
                f.create_dataset("prediction", data=pred_mask.astype(np.uint8), compression="gzip")

        if label_paths is not None:
            labels = _load_full_volume(label_paths[idx], "label")
            if labels.ndim == 3:
                labels = labels[..., np.newaxis]
            lbl_mask = labels[..., 0].astype(np.float32)
            dice = (2 * np.sum(pred_mask * lbl_mask) + 1e-5) / (
                np.sum(pred_mask) + np.sum(lbl_mask) + 1e-5
            )
            metrics.append({
                "volume": img_path.name,
                "dice": float(dice),
            })

    if metrics:
        metrics_path = Path(args.output_dir) / f"{split_name}_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
#                                CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised 3-D U-Net baseline")
    parser.add_argument(
        "--image-dir",
        default="./data/bioimage",
        help="Directory containing image HDF5 volumes (either flat or with train/val/test sub-folders)",
    )
    parser.add_argument(
        "--label-dir",
        default="./data/segmentation",
        help="Directory containing label HDF5 volumes (either flat or with train/val/test sub-folders)",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Sub-folder name for training volumes",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        help="Sub-folder name for validation volumes",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Sub-folder name for test volumes",
    )
    parser.add_argument("--output-dir", default="./unet_baseline_output", help="Directory to store weights and predictions")
    parser.add_argument("--patch-size", nargs=3, type=int, default=(128, 128, 128), help="Training patch size (HxWxD)")
    parser.add_argument("--inference-stride", nargs=3, type=int, default=(64, 64, 64), help="Sliding window stride (HxWxD)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--inference-batch-size", type=int, default=4, help="Number of patches evaluated at once during inference")
    parser.add_argument("--window-type", choices=["tukey", "hann", "gaussian"], default="tukey")
    parser.add_argument("--window-kwargs", default="", help="JSON encoded keyword arguments for the window function")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarisation threshold for predictions")
    parser.add_argument("--save-predictions", action="store_true", help="Persist predicted masks as HDF5 volumes")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of volumes reserved for validation when splitting flat directories")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of volumes reserved for testing when splitting flat directories")
    parser.add_argument("--split-seed", type=int, default=1337, help="Random seed for automatic train/val/test partitioning")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = (args.train_split, args.val_split, args.test_split)
    split_map = discover_split_paths(
        Path(args.image_dir),
        Path(args.label_dir),
        split_names,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
    )

    patch_shape = tuple(args.patch_size)
    model = train_unet(args, split_map, patch_shape)

    test_metrics = run_inference_on_split(
        model,
        args,
        args.test_split,
        split_map[args.test_split].images,
        label_paths=split_map[args.test_split].labels,
    )

    for split in split_names:
        if split == args.test_split:
            continue
        run_inference_on_split(
            model,
            args,
            split,
            split_map[split].images,
        )

    print("Test metrics:")
    for metric in test_metrics:
        print(metric)


if __name__ == "__main__":
    main()

