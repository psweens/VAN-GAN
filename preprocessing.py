# preprocess_hdf5.py
# ---------------------------------------------------------------------------
# Convert raw TIFF stacks → chunked-LZF HDF5 volumes for DatasetGen.
# ---------------------------------------------------------------------------
import os, shutil, random, multiprocessing, h5py
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import skimage.io as sk

from utils import (
    min_max_norm,
    check_nan,
    save_dict,
    load_dict,
    resize_volume,
    load_volume,
)


class DataPreprocessor:
    """
    Prepares three partitions (train / val / test) and writes each volume
    to a .h5 file with a single dataset:

        ─ partition A  →  /image     (float32, shape (X,Y[,Z],C))
        ─ partition B  →  /label     (float32, shape (X,Y[,Z],1))

    The layout matches the expected input of the HDF5-enabled DatasetGen.
    """

    # ─────────────────────────── init ───────────────────────────────
    def __init__(
        self,
        args=None,
        raw_path=None,
        main_dir=None,
        *,
        partition_id="",
        partition_filename=None,
        tiff_size=(600, 600, 700),
        target_size=(600, 600, 700),
        num_cores=multiprocessing.cpu_count() - 1,
    ):
        self.raw_path = raw_path
        self.main_dir = main_dir
        self.partition_id = partition_id            # "A" or "B"
        self.partition_filename = partition_filename or f"partition_{partition_id}.pkl"
        self.tiff_size = tiff_size
        self.target_size = target_size
        self.partition: dict[str, np.ndarray] = {}
        self.data_type = "float32"

        # runtime options
        self.preprocess_fn = None
        self.resize = False
        self.save_filtered = False

        # parallelism
        self.NUM_CORES = int(0.8 * num_cores)

        # model-related parameters (only if args supplied)
        if args is not None:
            self.DIMENSIONS = args.DIMENSIONS
            self.CHANNELS = args.CHANNELS
            # choose a reasonable default chunk shape ≈ patch size
            if self.DIMENSIONS == 2:
                self.chunk_shape = (
                    min(256, args.SUBVOL_PATCH_SIZE[0]),
                    min(256, args.SUBVOL_PATCH_SIZE[1]),
                    self.CHANNELS,
                )
            else:
                self.chunk_shape = (
                    min(128, args.SUBVOL_PATCH_SIZE[0]),
                    min(128, args.SUBVOL_PATCH_SIZE[1]),
                    min(16,  args.SUBVOL_PATCH_SIZE[2]),
                    self.CHANNELS if partition_id == "A" else 1,
                )
        else:  # fall-back chunk shape
            self.DIMENSIONS = 3
            self.CHANNELS = 1
            self.chunk_shape = None

    # ───────────────────────── partition helpers ────────────────────
    def split_dataset(self):
        files = os.listdir(self.raw_path)
        random.shuffle(files)

        n = len(files)
        train, test = np.split(files, [int(0.9 * n)])
        train, val  = np.split(train, [int(0.8 * len(train))])

        self.partition = {
            "training":   np.array(train, dtype=object),
            "validation": np.array(val,   dtype=object),
            "testing":    np.array(test,  dtype=object),
        }

    def save_partition(self, save_path: str):
        if not save_path:
            raise ValueError("save_path must be provided")

        def _replace(fname, split):
            stem, _ = os.path.splitext(fname)
            return os.path.join(save_path, f"{split}{self.partition_id}", stem + ".h5")

        new_part = {
            split: np.array([_replace(f, lab) for f in arr], dtype=object)
            for (split, arr), lab in zip(
                self.partition.items(), ("train", "val", "test")
            )
        }
        save_dict(new_part, os.path.join(save_path, self.partition_filename))
        self.partition = new_part

    # ───────────────────────── public API ───────────────────────────
    def preprocess(
        self,
        preprocess_fn=None,
        resize=False,
        save_filtered=False,
    ):
        print(f"*** Pre-processing partition {self.partition_id} ***")

        self.preprocess_fn = preprocess_fn
        self.resize = resize
        self.save_filtered = save_filtered

        # ensure output folders exist
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(self.main_dir, f"{split}{self.partition_id}"),
                        exist_ok=True)
        if save_filtered:
            os.makedirs(os.path.join(self.main_dir, "filtered",
                                      f"{split}{self.partition_id}"), exist_ok=True)

        self.split_dataset()

        # --- parallel processing of TIFF stacks --------------------
        def _par(func, files, label):
            return Parallel(n_jobs=self.NUM_CORES, verbose=50)(
                delayed(func)(file=f, split_label=label) for f in files
            )

        _par(self._process_one, self.partition["training"],   "train")
        _par(self._process_one, self.partition["validation"], "val")
        _par(self._process_one, self.partition["testing"],    "test")

        self.save_partition(self.main_dir)

    # ───────────────────────── single-file worker ───────────────────
    def _process_one(self, file, *, split_label):
        # -------- load & normalise (unchanged) -------------------
        stack = load_volume(os.path.join(self.raw_path, file),
                            datatype=self.data_type,
                            normalise=True)
        if self.DIMENSIONS == 3:
            stack = np.transpose(stack, (1, 2, 0))

        if self.preprocess_fn is not None:
            stack = self.preprocess_fn(stack)

        if self.resize and (self.tiff_size != self.target_size):
            stack = resize_volume(stack, self.target_size).astype(self.data_type)
            if self.partition_id == "B":
                stack = np.clip(stack, 0, 255)

        if self.partition_id == "B":
            stack = min_max_norm(stack)
            mode, _ = stats.mode(stack, axis=None)
            if mode == 1:
                stack = np.abs(stack - 1.0)
            stack = (stack - 0.5) / 0.5
            stack[stack < 0] = -1.0
            stack[stack >= 0] = 1.0

        if check_nan(stack):
            print("NaN detected in", file)
            return

        # ------------- add channel dim where needed --------------
        if self.partition_id == "B":
            stack = stack[..., None]                       # label → 1-channel
            dset_name = "label"
        else:                                              # imaging
            if self.CHANNELS == 1:
                stack = stack[..., None]
            dset_name = "image"

        # ------------- dynamic chunk shape  ----------------------
        # aim for ~128 kB chunks but never bigger than the data shape
        def _auto_chunks(shape, target_bytes=128 * 1024, dtype=np.float32):
            # start with full shape then iteratively halve longest axes
            chunk = list(shape)
            bytes_per_elem = np.dtype(dtype).itemsize
            while np.prod(chunk) * bytes_per_elem > target_bytes:
                # halve the largest dimension (>1)
                idx = int(np.argmax(chunk))
                if chunk[idx] > 1:
                    chunk[idx] = (chunk[idx] + 1) // 2
                else:
                    break
            return tuple(chunk)

        chunk_shape = _auto_chunks(stack.shape)

        # ------------- write HDF5  -------------------------------
        dst_dir = os.path.join(self.main_dir, f"{split_label}{self.partition_id}")
        os.makedirs(dst_dir, exist_ok=True)
        h5_path = os.path.join(dst_dir, os.path.splitext(file)[0] + ".h5")

        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                dset_name,
                data=stack.astype("float32"),
                chunks=chunk_shape,
                compression="lzf",
                shuffle=True,
            )

        # ----- save filtered PNG preview (optional) ---------------
        if self.save_filtered:
            out_png = os.path.join(
                self.main_dir, "filtered",
                f"{split_label}{self.partition_id}",
                os.path.splitext(file)[0] + ".tiff",
            )
            if self.DIMENSIONS == 3:
                sk.imsave(
                    out_png,
                    (np.transpose(stack, (2, 0, 1)) * 127.5 + 127.5).astype("uint8"),
                    bigtiff=False,
                    check_contrast=False,
                )
            else:
                sk.imsave(
                    out_png, (stack * 127.5 + 127.5).astype("uint8"),
                    bigtiff=False, check_contrast=False
                )

    # ───────────────────────── util: load existing partition ───────
    def load_partition(self, file_path):
        print(f"*** Loading dataset {self.partition_id} partition ***")
        self.partition = load_dict(file_path)

    # ───────────────────────── util: process unseen data ───────────
    def process_new_data(
        self,
        current_path,
        new_path,
        *,
        tiff_size=None,
        target_size=None,
        preprocess_fn=None,
        resize=False,
    ):
        self.raw_path = current_path
        self.main_dir = new_path
        self.tiff_size = tiff_size or self.tiff_size
        self.target_size = target_size or self.target_size
        self.preprocess_fn = preprocess_fn
        self.resize = resize
        self.save_filtered = False

        os.makedirs(new_path, exist_ok=True)
        for f in os.listdir(current_path):
            self._process_one(file=f, split_label="new")
