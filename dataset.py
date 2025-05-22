# dataset_gen.py
# ─────────────────────────────────────────────────────────────────────────────
# High-throughput HDF5 loader for 2-D / 3-D microscopy GANs
# ─────────────────────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from skimage import io


class DatasetGen:
    """
    tf.data-based pipeline that reads a random patch *directly* from a chunked,
    LZF-compressed HDF5 volume.  Designed for one accepted patch per file.
    """

    # ───────────────────────── constructor ────────────────────────────
    def __init__(
            self,
            args,
            imaging_paths: dict,  # {"training": [...], "validation": [...]}
            segmentation_paths: dict,
            strategy: tf.distribute.Strategy,
            *,
            otf_imaging=None,
            surface_illumination: bool = False,
            semi_supervised_dir: str = None,
            plot_samples: bool = True,
    ):
        # ------- keep references to user flags -------
        self.args = args
        self.strategy = strategy
        self.DIMENSIONS = args.DIMENSIONS
        self.GLOBAL_BATCH_SIZE = args.GLOBAL_BATCH_SIZE
        self.surface_illumination = surface_illumination
        self.otf_imaging = otf_imaging

        # ------- shapes -------------------------------------------------
        if self.DIMENSIONS == 2:
            self.imaging_patch_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                args.CHANNELS,
            )
            self.segmentation_patch_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                1,
            )
        else:  # 3-D
            self.imaging_patch_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                args.SUBVOL_PATCH_SIZE[2],
                args.CHANNELS,
            )
            self.segmentation_patch_shape = (
                args.SUBVOL_PATCH_SIZE[0],
                args.SUBVOL_PATCH_SIZE[1],
                args.SUBVOL_PATCH_SIZE[2],
                1,
            )

        # ─── pick cropping strategy at start-up ────────────────────────────
        self._use_small_seg_method = (
                tuple(self.segmentation_patch_shape[:3]) == (64, 64, 64)
        )
        self._use_small_seg_method = True
        if self._use_small_seg_method:
            print('Applying feature cropping for segmentation domain.')
        else:
            print('Applying random cropping for segmentation domain.')

        # ------- semi-supervised pair logic ----------------------------
        self.semi_supervised = semi_supervised_dir is not None
        self.semi_supervised_dir = semi_supervised_dir or ""
        if self.semi_supervised:
            self.ss_patch_shape = (
                self.segmentation_patch_shape[0],
                self.segmentation_patch_shape[1],
                2 * self.segmentation_patch_shape[2]  # concat along Z
                if self.DIMENSIONS == 3
                else 2,
                1,
            )
        n_spatial_dims = 2 if self.is_2d() else 3
        self._label_crop_sz = (
            (self.ss_patch_shape if self.semi_supervised else self.segmentation_patch_shape)
            [:n_spatial_dims]
        )

        # ---------- file lists -----------------------------------------
        self.imaging_paths = imaging_paths
        self.segmentation_paths = segmentation_paths

        # ---------- tf.data options ------------------------------------
        ds_opts = tf.data.Options()
        ds_opts.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

        # ------------------------------------------------------------------
        # build individual domain datasets
        # ------------------------------------------------------------------
        def build_imaging_ds(paths: list[str]):
            return (
                tf.data.Dataset.from_tensor_slices(paths)
                .shuffle(len(paths))
                .repeat()
                .map(
                    lambda p: tf.numpy_function(
                        DatasetGen._load_im_patch_h5,
                        [p, self.imaging_patch_shape, self.DIMENSIONS],
                        tf.float32,
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .map(self.process_imaging_domain, tf.data.AUTOTUNE)
                .with_options(ds_opts)
            )

        def build_segmentation_ds(paths: list[str]):
            # choose which post-processing function to use
            proc_fn = (self.seg_feature_crop
                       if self._use_small_seg_method
                       else self.seg_random_crop)

            return (
                tf.data.Dataset.from_tensor_slices(paths)
                .shuffle(len(paths))
                .repeat()
                .map(
                    lambda p: tf.numpy_function(
                        DatasetGen._load_seg_patch_h5,
                        [
                            p,
                            self.ss_patch_shape if self.semi_supervised
                            else self.segmentation_patch_shape,
                            self.DIMENSIONS,
                            0.8,
                            self.semi_supervised,
                            self.semi_supervised_dir.encode("utf-8"),
                        ],
                        tf.float32,
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .map(proc_fn, num_parallel_calls=tf.data.AUTOTUNE)
                .with_options(ds_opts)
            )

        with self.strategy.scope():
            im_train = build_imaging_ds(imaging_paths["training"])
            im_val = build_imaging_ds(imaging_paths["validation"])

            seg_train = build_segmentation_ds(segmentation_paths["training"])
            seg_val = build_segmentation_ds(segmentation_paths["validation"])

            # ----- zip, batch, optional OTF imaging, prefetch -------------
            def pack(a, b):
                ds = tf.data.Dataset.zip((a, b))
                ds = ds.batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True)
                if otf_imaging is not None:
                    ds = ds.map(
                        lambda x, y: (self.otf_imaging(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False,
                    )
                return ds.prefetch(tf.data.AUTOTUNE)

            self.train_dataset = strategy.experimental_distribute_dataset(
                pack(im_train, seg_train)
            )
            self.val_dataset = strategy.experimental_distribute_dataset(
                pack(im_val, seg_val)
            )

        # ---------------- sanity-check figures ----------------------------
        if plot_samples:
            self._plot_sample_dataset()

        # ------------------------------------------------------------------
        #  Full-volume validation datasets (used only by GanMonitor & friends)
        # ------------------------------------------------------------------
        def _load_full_volume_h5(path_bytes, dset_name=b"image"):
            """Read the *entire* volume from an .h5 file into a float32 array."""

            path = path_bytes.decode("utf-8")
            with h5py.File(path, "r") as f:
                vol = f[dset_name.decode("utf-8")][...].astype(np.float32)
            return vol

        # ------- imaging (index, volume) ----------------------------------
        im_val_paths = tf.data.Dataset.from_tensor_slices(
            self.imaging_paths["validation"]
        ).enumerate()  # (index, path)

        self.imaging_val_full_vol_data = im_val_paths.map(
            lambda i, p: (
                tf.numpy_function(_load_full_volume_h5, [p, b"image"], tf.float32),
                tf.cast(i, tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        # ------- segmentation (index, volume) -----------------------------
        seg_val_paths = tf.data.Dataset.from_tensor_slices(
            self.segmentation_paths["validation"]
        ).enumerate()

        self.segmentation_val_full_vol_data = seg_val_paths.map(
            lambda i, p: (
                tf.numpy_function(_load_full_volume_h5, [p, b"label"], tf.float32),
                tf.cast(i, tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    # ==========================================================================
    #                     ----------  static I/O helpers  ----------
    # ==========================================================================

    @staticmethod
    def _open_h5(path_str: str):
        """Open HDF5 in SWMR-safe read-only mode with a 4 MiB chunk cache."""
        return h5py.File(
            path_str,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=4 << 20,
        )

    @staticmethod
    def _random_indices(vol_shape, patch_shape):
        return tuple(
            random.randrange(0, int(s) - int(p))
            for s, p in zip(vol_shape, patch_shape)
        )

    # ---------------- imaging -------------------------------------------------
    @staticmethod
    def _load_im_patch_h5(path, patch_shape, ndim):
        """Return one random imaging patch from `/image`."""
        path = path.decode()
        with DatasetGen._open_h5(path) as f:
            ds = f["image"]
            if ndim == 2:
                x0, y0 = DatasetGen._random_indices(ds.shape[:2], patch_shape[:2])
                patch = ds[x0:x0 + patch_shape[0],
                        y0:y0 + patch_shape[1], :]
            else:
                x0, y0, z0 = DatasetGen._random_indices(ds.shape[:3], patch_shape[:3])
                patch = ds[x0:x0 + patch_shape[0],
                        y0:y0 + patch_shape[1],
                        z0:z0 + patch_shape[2], :]
            return np.asarray(patch, dtype=np.float32)

    # ---------------- segmentation (+ optional paired imaging) ---------------
    @staticmethod
    def _load_seg_patch_h5(
            path,
            patch_shape,
            ndim,
            thresh: float,
            semi_supervised: bool,
            ss_dir: bytes,
            max_tries: int = 10,
    ):
        path = path.decode()
        with DatasetGen._open_h5(path) as f:
            ds = f["label"]
            for _ in range(max_tries):
                if ndim == 2:
                    x0, y0 = DatasetGen._random_indices(ds.shape[:2], patch_shape[:2])
                    patch = ds[x0:x0 + patch_shape[0],
                            y0:y0 + patch_shape[1], :]
                else:
                    x0, y0, z0 = DatasetGen._random_indices(ds.shape[:3], patch_shape[:3])
                    patch = ds[x0:x0 + patch_shape[0],
                            y0:y0 + patch_shape[1],
                            z0:z0 + patch_shape[2], :]
                if patch.max() > thresh:
                    break
            patch = np.asarray(patch, np.float32)

        if semi_supervised:
            with DatasetGen._open_h5(
                    os.path.join(ss_dir.decode(), os.path.basename(path))
            ) as f_im:
                im_ds = f_im["image"]
                if ndim == 2:
                    paired = im_ds[x0:x0 + patch_shape[0],
                             y0:y0 + patch_shape[1], :]
                else:
                    paired = im_ds[x0:x0 + patch_shape[0],
                             y0:y0 + patch_shape[1],
                             z0:z0 + patch_shape[2], :]
                patch = np.concatenate([patch, paired], axis=2)  # concat along Z
        return patch

    @staticmethod
    def _maybe_skip(target_shape, apply_fn, skip_rate=0.1):
        """
        With probability skip_rate returns a patch full of –1’s of shape target_shape,
        otherwise calls apply_fn() to produce a “real” crop.
        """
        # draw a scalar U~Uniform[0,1)
        skip = tf.less(tf.random.uniform([], 0, 1), skip_rate)
        # build a “no‐feature” tensor of the right shape
        no_feat = tf.fill(tf.constant(target_shape, tf.int32),
                          tf.constant(-1, dtype=tf.float32))
        return tf.cond(skip, lambda: no_feat, apply_fn)


    # ==========================================================================
    #                        ------ augmentations ------                       #
    # ==========================================================================

    def is_2d(self):
        return self.DIMENSIONS == 2

    @tf.function
    def random_intensity_augmentation(
            self, img, brightness_delta=0.2, contrast_range=(0.7, 1.3)
    ):
        img = tf.image.random_brightness(img, brightness_delta)
        return tf.image.random_contrast(img, *contrast_range)

    @tf.function
    def random_rotate(self, img, preserve_z_axis=False):
        if self.is_2d():
            img = tf.image.random_flip_left_right(img)
            return tf.image.rot90(img, k=tf.random.uniform((), 0, 4, tf.int32))
        else:
            return self._random_rotate_3d(img, preserve_z_axis)

    @tf.function
    def _random_rotate_3d(self, img, preserve_z_axis):
        preserve_z_axis = tf.convert_to_tensor(preserve_z_axis, tf.bool)
        do_rot = tf.greater(tf.random.uniform(()), 0.5)

        def _rot():
            def _rot_z():
                k = tf.random.uniform((), 0, 4, tf.int32)
                t = tf.transpose(img, [2, 1, 0, 3])
                return tf.transpose(tf.image.rot90(t, k), [2, 1, 0, 3])

            def _rot_rand():
                axis = tf.random.uniform((), 0, 3, tf.int32)
                k = tf.random.uniform((), 1, 4, tf.int32)

                def rx():
                    t = tf.transpose(img, [0, 2, 1, 3])
                    return tf.transpose(tf.image.rot90(t, k), [0, 2, 1, 3])

                def ry():
                    t = tf.transpose(img, [1, 2, 0, 3])
                    return tf.transpose(tf.image.rot90(t, k), [2, 0, 1, 3])

                def rz():
                    t = tf.transpose(img, [2, 1, 0, 3])
                    return tf.transpose(tf.image.rot90(t, k), [2, 1, 0, 3])

                return tf.switch_case(axis, [rx, ry, rz])

            return tf.cond(preserve_z_axis, _rot_z, _rot_rand)

        return tf.cond(do_rot, _rot, lambda: tf.identity(img))

    # ---------- per-domain wrappers ------------------------------------------
    def process_imaging_domain(self, img):
        img = self.random_rotate(img, self.surface_illumination)
        return self.random_intensity_augmentation(img)

    # ----------------------------------------------------------------------
    #   two alternative cropping strategies for the SEGMENTATION domain
    # ----------------------------------------------------------------------
    @tf.function
    def seg_random_crop(self, img):
        # pick the full target shape (including channels)
        target_shape = (self.ss_patch_shape
                        if self.semi_supervised
                        else self.segmentation_patch_shape)

        # the branch that actually does up to 10 tries
        def _do_loop():
            # fresh draw
            def _rand_draw(x):
                return tf.image.random_crop(x, size=target_shape)

            init = _rand_draw(img)

            # keep drawing until we hit a “good” seg (max >= .8)
            def _cond(patch, orig):
                if self.semi_supervised:
                    axis = 2 if self.is_2d() else 3
                    seg, _ = tf.split(patch, num_or_size_splits=2, axis=axis)
                    return tf.less(tf.reduce_max(seg), 0.8)
                else:
                    return tf.less(tf.reduce_max(patch), 0.8)

            def _body(_, orig):
                return _rand_draw(orig), orig

            # unknown spatial dims → declare invariants
            rank = len(target_shape)
            inv = tf.TensorShape([None] * rank)

            patch, _ = tf.while_loop(
                _cond, _body,
                loop_vars=[init, img],
                shape_invariants=[inv, inv],
                maximum_iterations=10,
            )
            return self.random_rotate(patch, self.surface_illumination)

        return self._maybe_skip(target_shape, _do_loop)

    @tf.function
    def seg_feature_crop(self, img):
        # spatial dims only, no channel
        spatial = (self.segmentation_patch_shape[:2]
                   if self.is_2d()
                   else self.segmentation_patch_shape[:3])
        crop_sz = tf.constant(spatial, tf.int32)

        # the branch that crops around a positive voxel
        def _do_centered():
            pos = tf.where(img > 0.)

            def _no_pos():
                # fallback if no positives at all
                return img

            def _has_pos():
                rnd = tf.cast(tf.random.shuffle(pos)[0][:len(spatial)], tf.int32)
                vol = tf.shape(img)[:len(spatial)]
                start = tf.minimum(tf.maximum(0, rnd - crop_sz // 2), vol - crop_sz)
                end = start + crop_sz

                if self.is_2d():
                    patch = img[start[0]:end[0], start[1]:end[1], :]
                else:
                    patch = img[start[0]:end[0],
                            start[1]:end[1],
                            start[2]:end[2], :]
                return self.random_rotate(patch, self.surface_illumination)

            return tf.cond(tf.equal(tf.size(pos), 0), _no_pos, _has_pos)

        # the final shape *including* channels
        target_shape = self.segmentation_patch_shape
        return self._maybe_skip(target_shape, _do_centered)

    # ==========================================================================
    #                         ----- visualisation -----                         #
    # ==========================================================================

    def _plot_sample_dataset(self, nfig=None):
        per_im, per_seg = next(iter(self.train_dataset))
        im = self.strategy.experimental_local_results(per_im)[0][0].numpy()
        seg = self.strategy.experimental_local_results(per_seg)[0][0].numpy()

        if self.otf_imaging is not None:
            im = self.otf_imaging(im[None])[0].numpy()

        if self.semi_supervised:
            z_sep = self.segmentation_patch_shape[2]
            seg_only = seg[:, :, :z_sep, 0] if self.DIMENSIONS == 3 else seg[:, :, 0, 0]
            paired = seg[:, :, z_sep:, 0] if self.DIMENSIONS == 3 else seg[:, :, 1, 0]
        else:
            seg_only, paired = (seg[:, :, :, 0] if self.DIMENSIONS == 3 else seg), None

        nfig = 1 if self.is_2d() else (nfig or 6)
        cols = 3 if paired is not None else 2

        # ---- XY figure ----------------------------------------------------
        fig1, ax = plt.subplots(nfig + 1, cols, figsize=(cols * 4, (nfig + 1) * 3))
        for j in range(nfig):
            z = j * (seg_only.shape[2] // nfig) if self.DIMENSIONS == 3 else 0
            ax[j, 0].imshow(im[:, :, z] if self.DIMENSIONS == 3 else im,
                            cmap="gray", vmin=-1, vmax=1)
            ax[j, 1].imshow(seg_only[:, :, z] if self.DIMENSIONS == 3 else seg_only,
                            cmap="gray", vmin=-1, vmax=1)
            if paired is not None:
                ax[j, 2].imshow(paired[:, :, z] if self.DIMENSIONS == 3 else paired,
                                cmap="gray", vmin=-1, vmax=1)

        ax[nfig, 0].hist(im.ravel(), 256, (-1, 1), fc="k", ec="k", density=True)
        ax[nfig, 1].hist(seg_only.ravel(), 256, (-1, 1), fc="k", ec="k", density=True)
        if paired is not None:
            ax[nfig, 2].hist(paired.ravel(), 256, (-1, 1), fc="k", ec="k", density=True)

        ax[0, 0].set_title("Imaging (XY)");
        ax[0, 1].set_title("Segmentation (XY)")
        if paired is not None: ax[0, 2].set_title("Paired imaging (XY)")
        plt.tight_layout();
        plt.savefig("./GANMonitor/XY_Dataset_Sample.png");
        plt.show()

        # ---- XZ figure (3-D only) -----------------------------------------
        if self.DIMENSIONS == 3:
            fig2, ax = plt.subplots(nfig, cols, figsize=(cols * 4, nfig * 3))
            for j in range(nfig):
                y = j * (seg_only.shape[1] // nfig)
                ax[j, 0].imshow(im[:, y, :], cmap="gray", vmin=-1, vmax=1)
                ax[j, 1].imshow(seg_only[:, y, :], cmap="gray", vmin=-1, vmax=1)
                if paired is not None:
                    ax[j, 2].imshow(paired[:, y, :], cmap="gray", vmin=-1, vmax=1)

            ax[0, 0].set_title("Imaging (XZ)");
            ax[0, 1].set_title("Segmentation (XZ)")
            if paired is not None: ax[0, 2].set_title("Paired imaging (XZ)")
            plt.tight_layout();
            plt.savefig("./GANMonitor/XZ_Dataset_Sample.png");
            plt.show()
