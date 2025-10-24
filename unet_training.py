from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import tensorflow as tf
import numpy as np
import h5py
import glob

from custom_callback import GanMonitor
from res_unet_model import ResUNet


@dataclass
class UnetTrainingConfig:
    epochs: int
    steps_per_epoch: int
    validation_steps: int
    learning_rate: float
    output_dir: str


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)


def get_callbacks(output_dir: str) -> Iterable[tf.keras.callbacks.Callback]:
    callbacks = []
    ckpt_dir = os.path.join(output_dir, "unet_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "weights_{epoch:03d}.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_dice_coefficient",
            mode="max",
        )
    )
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            patience=20,
            mode="max",
            restore_best_weights=True,
        )
    )
    log_dir = os.path.join(output_dir, "unet_logs")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    return callbacks


def build_unet(args) -> tf.keras.Model:
    input_shape = (
        args.SUBVOL_PATCH_SIZE[0],
        args.SUBVOL_PATCH_SIZE[1],
        args.SUBVOL_PATCH_SIZE[2],
        args.CHANNELS,
    )
    model = ResUNet(
        input_shape=input_shape,
        dim=args.DIMENSIONS,
        output_activation="sigmoid",
        filters=32,
        num_layers=4,
        use_input_noise=False,
    )
    return model


@dataclass
class UnetTrainingResult:
    model: tf.keras.Model
    history: tf.keras.callbacks.History
    best_checkpoint: str


def train_unet(
    args,
    strategy: tf.distribute.Strategy,
    dataset,
    *,
    config: Optional[UnetTrainingConfig] = None,
) -> UnetTrainingResult:
    if config is None:
        config = UnetTrainingConfig(
            epochs=args.EPOCHS,
            steps_per_epoch=dataset.train_steps,
            validation_steps=dataset.val_steps,
            learning_rate=args.INITIAL_LR,
            output_dir=args.output_dir,
        )

    with strategy.scope():
        model = build_unet(args)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [dice_coefficient, tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.Precision(name="precision")]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = list(get_callbacks(config.output_dir))

    history = model.fit(
        dataset.train_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=dataset.val_dataset,
        validation_steps=config.validation_steps,
        callbacks=callbacks,
    )

    ckpt_dir = os.path.join(config.output_dir, "unet_checkpoints")
    checkpoint_pattern = os.path.join(ckpt_dir, "weights_*.weights.h5")
    candidates = glob.glob(checkpoint_pattern)
    if not candidates:
        raise FileNotFoundError(
            "No checkpoints were saved during U-Net training; expected to find files matching "
            f"{checkpoint_pattern}."
        )
    best_checkpoint = max(candidates, key=os.path.getmtime)

    return UnetTrainingResult(model=model, history=history, best_checkpoint=best_checkpoint)


def load_unet_from_checkpoint(
    args,
    strategy: tf.distribute.Strategy,
    checkpoint_path: str,
) -> tf.keras.Model:
    """Rebuild the 3D U-Net architecture and load weights from a checkpoint."""

    with strategy.scope():
        model = build_unet(args)
        model.load_weights(checkpoint_path)
    return model


def _scale_volume(volume: np.ndarray, args) -> np.ndarray:
    min_val = np.float32(args.MIN_PIXEL_VALUE)
    max_val = np.float32(args.MAX_PIXEL_VALUE)
    clipped = np.clip(volume, min_val, max_val)
    scaled = (clipped - min_val) / (max_val - min_val + 1e-8)
    return scaled * 2.0 - 1.0


def _dice_score(pred: np.ndarray, truth: np.ndarray, smooth: float = 1e-6) -> float:
    pred_flat = pred.reshape(-1)
    truth_flat = truth.reshape(-1)
    intersection = np.dot(pred_flat, truth_flat)
    denominator = pred_flat.sum() + truth_flat.sum()
    return float((2.0 * intersection + smooth) / (denominator + smooth))


def _resolve_stride(
    stride: Optional[Sequence[int]],
    patch_shape: Sequence[int],
    dims: int,
) -> Tuple[int, ...]:
    if stride is None:
        return tuple(int(s) for s in patch_shape[:dims])
    if isinstance(stride, Sequence):
        if len(stride) == 1:
            stride_val = int(stride[0])
            return tuple([stride_val] * dims)
        if len(stride) != dims:
            raise ValueError(
                f"Stride length {len(stride)} does not match spatial dimensions {dims}."
            )
        return tuple(int(s) for s in stride)
    return tuple([int(stride)] * dims)


class _EmptyDataset:
    imaging_val_full_vol_data: Sequence[str] = ()
    segmentation_val_full_vol_data: Sequence[str] = ()


def segment_volumes_with_unet(
    args,
    model: tf.keras.Model,
    volume_paths: Sequence[str],
    output_dir: str,
    *,
    stride: Optional[Sequence[int]] = None,
    threshold: float = 0.5,
    label_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Generate segmentation masks for the provided volumes using the supplied model.

    The stitched probability volumes are returned for metric computation and saved
    alongside TIFF exports produced by :meth:`GanMonitor.stitch_subvolumes`.
    """

    if not volume_paths:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    spatial_dims = args.DIMENSIONS

    # `stitch_subvolumes` expects the full input tensor shape including batch axis
    subvol_shape = args.INPUT_IMG_SIZE

    stitch_stride: Optional[Tuple[int, int, int]]
    if stride is None:
        stitch_stride = None
    else:
        patch_shape = args.INPUT_IMG_SIZE[1:1 + spatial_dims]
        resolved_stride = _resolve_stride(stride, patch_shape, spatial_dims)
        if spatial_dims == 2:
            stitch_stride = tuple(resolved_stride) + (resolved_stride[-1],)
        else:
            stitch_stride = tuple(resolved_stride)

    stitch_helper = GanMonitor(
        args,
        dataset=_EmptyDataset(),
        imaging_val_data=[],
        segmentation_val_data=[],
        process_imaging_domain=None,
        surface_illumination=getattr(args, "SURFACE_ILLUMINATION", False),
    )

    results: Dict[str, Dict[str, Optional[float]]] = {}

    for vol_path in volume_paths:
        with h5py.File(vol_path, "r") as f:
            volume = np.asarray(f["image"], dtype=np.float32)

        scaled_volume = _scale_volume(volume, args)

        scaled_input = scaled_volume[..., np.newaxis]

        base = os.path.splitext(os.path.basename(vol_path))[0]

        stitched_prediction = stitch_helper.stitch_subvolumes(
            lambda batch: model(batch, training=False),
            scaled_input,
            subvol_shape,
            name=base,
            stride=stitch_stride,
            output_path=output_dir,
            complete=True,
            save_output=True,
            return_prediction=True,
        )

        probability_map = np.squeeze(stitched_prediction, axis=-1)
        probability_map = np.clip(probability_map, 0.0, 1.0)
        binary_prediction = (probability_map >= threshold).astype(np.uint8)

        output_path = os.path.join(output_dir, f"{base}_prediction.h5")
        with h5py.File(output_path, "w") as out_f:
            out_f.create_dataset("probability", data=probability_map, compression="gzip")
            out_f.create_dataset("prediction", data=binary_prediction, compression="gzip")

        dice_value: Optional[float] = None
        if label_map is not None and base in label_map:
            with h5py.File(label_map[base], "r") as label_file:
                label_volume = np.asarray(label_file["label"], dtype=np.float32)
            if label_volume.min() < 0:
                label_volume = (label_volume > 0).astype(np.float32)
            dice_value = _dice_score(binary_prediction.astype(np.float32), label_volume)

        results[vol_path] = {
            "prediction_path": output_path,
            "tiff_path": os.path.join(output_dir, f"{base}.tiff"),
            "dice": dice_value,
        }

    return results
