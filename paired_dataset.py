import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf


def _pair_files(im_paths: Sequence[str], seg_paths: Sequence[str]) -> List[Tuple[str, str]]:
    """Match imaging and segmentation files based on their basename."""
    seg_lookup = {os.path.basename(p): p for p in seg_paths}
    pairs: List[Tuple[str, str]] = []
    missing = []
    for im_path in im_paths:
        key = os.path.basename(im_path)
        seg_path = seg_lookup.get(key)
        if seg_path is None:
            missing.append(key)
            continue
        pairs.append((im_path, seg_path))
    if missing:
        print(
            "[PairedDatasetGen] Warning: could not find matching segmentation files for",
            len(missing),
            "volumes. Examples:",
            missing[:5],
        )
    if not pairs:
        raise ValueError("No paired imaging/segmentation files were found.")
    return pairs


def _random_indices(volume_shape: Sequence[int], patch_shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(
        random.randrange(0, int(s) - int(p))
        for s, p in zip(volume_shape, patch_shape)
    )


def _load_pair_patch(
    im_path: bytes,
    seg_path: bytes,
    patch_shape: Sequence[int],
    ndim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a random paired patch from the imaging and label volumes."""
    with h5py.File(im_path.decode("utf-8"), "r") as f_im:
        im_vol = f_im["image"]
        if ndim == 2:
            x0, y0 = _random_indices(im_vol.shape[:2], patch_shape[:2])
            patch_im = im_vol[x0 : x0 + patch_shape[0], y0 : y0 + patch_shape[1], :]
        else:
            x0, y0, z0 = _random_indices(im_vol.shape[:3], patch_shape[:3])
            patch_im = im_vol[
                x0 : x0 + patch_shape[0],
                y0 : y0 + patch_shape[1],
                z0 : z0 + patch_shape[2],
                :,
            ]

    with h5py.File(seg_path.decode("utf-8"), "r") as f_seg:
        seg_vol = f_seg["label"]
        if ndim == 2:
            patch_seg = seg_vol[x0 : x0 + patch_shape[0], y0 : y0 + patch_shape[1], :]
        else:
            patch_seg = seg_vol[
                x0 : x0 + patch_shape[0],
                y0 : y0 + patch_shape[1],
                z0 : z0 + patch_shape[2],
                :,
            ]

    patch_im = np.asarray(patch_im, dtype=np.float32)
    patch_seg = np.asarray(patch_seg, dtype=np.float32)

    # Convert the segmentation labels from {-1, 1} to {0, 1} if needed.
    if patch_seg.min() < 0:
        patch_seg = (patch_seg > 0).astype(np.float32)

    return patch_im, patch_seg


class PairedDatasetGen:
    """Create paired imaging/segmentation datasets for supervised training."""

    def __init__(
        self,
        args,
        imaging_paths: Dict[str, Iterable[str]],
        segmentation_paths: Dict[str, Iterable[str]],
        *,
        strategy: tf.distribute.Strategy,
        otf_imaging=None,
    ) -> None:
        self.args = args
        self.strategy = strategy
        self.otf_imaging = otf_imaging
        self.ndim = args.DIMENSIONS

        if self.ndim == 2:
            self.patch_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                args.CHANNELS,
            )
            self.label_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                1,
            )
        else:
            self.patch_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                args.SUBVOL_PATCH_SIZE[2],
                args.CHANNELS,
            )
            self.label_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                args.SUBVOL_PATCH_SIZE[2],
                1,
            )

        self.train_pairs = _pair_files(
            imaging_paths.get("training", []),
            segmentation_paths.get("training", []),
        )
        self.val_pairs = _pair_files(
            imaging_paths.get("validation", []),
            segmentation_paths.get("validation", []),
        )

        self.train_steps = max(1, len(self.train_pairs) // args.GLOBAL_BATCH_SIZE)
        self.val_steps = max(1, len(self.val_pairs) // args.GLOBAL_BATCH_SIZE)

        self.train_dataset = self._build_dataset(self.train_pairs, training=True)
        self.val_dataset = self._build_dataset(self.val_pairs, training=False)

    def _build_dataset(self, pairs: Sequence[Tuple[str, str]], *, training: bool) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(pairs)
        if training:
            dataset = dataset.shuffle(len(pairs))
        dataset = dataset.repeat()

        def _load(im_path, seg_path):
            im, seg = tf.numpy_function(
                func=_load_pair_patch,
                inp=[im_path, seg_path, self.patch_shape, self.ndim],
                Tout=(tf.float32, tf.float32),
            )
            im.set_shape(self.patch_shape)
            seg.set_shape(self.label_shape)
            if self.otf_imaging is not None:
                im = self.otf_imaging(im)
            else:
                im = self._scale_imaging(im)
            if training:
                im, seg = self._random_flip(im, seg)
            return im, seg

        dataset = dataset.map(
            _load,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not training,
        )

        dataset = dataset.batch(self.args.GLOBAL_BATCH_SIZE, drop_remainder=True)
        return dataset.prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def _random_flip(im: tf.Tensor, seg: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        def _flip_axis(axis: int, tensor: tf.Tensor) -> tf.Tensor:
            return tf.reverse(tensor, axis=[axis])

        if tf.random.uniform(()) > 0.5:
            im = _flip_axis(0, im)
            seg = _flip_axis(0, seg)
        if tf.random.uniform(()) > 0.5:
            im = _flip_axis(1, im)
            seg = _flip_axis(1, seg)
        if im.shape.rank == 4 and tf.random.uniform(()) > 0.5:
            im = _flip_axis(2, im)
            seg = _flip_axis(2, seg)
        return im, seg

    def _scale_imaging(self, tensor: tf.Tensor) -> tf.Tensor:
        min_val = tf.cast(self.args.MIN_PIXEL_VALUE, tf.float32)
        max_val = tf.cast(self.args.MAX_PIXEL_VALUE, tf.float32)
        tensor = tf.clip_by_value(tensor, min_val, max_val)
        tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
        return tensor * 2.0 - 1.0
