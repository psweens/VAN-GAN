import os
from collections.abc import Iterable
from typing import Any


def _is_sequence(obj: Any) -> bool:
    """
    Return True if *obj* behaves like a (non-string) sequence / iterable
    that we can iterate over, e.g. list, tuple, numpy.ndarray, etc.

    Strings / bytes / os.PathLike instances are *not* considered a sequence
    for the purpose of this function because in the context of this module
    they represent single paths.
    """
    # Treat path-like values as atomic objects, not sequences of paths
    if isinstance(obj, (str, bytes, os.PathLike)):
        return False

    # A 'true' sequence/iterable must be iterable *and* have a length
    return isinstance(obj, Iterable) and hasattr(obj, "__len__")


def epoch_sweep(
    args,
    vangan_model,
    plotter,
    test_path="",
    start=100,
    end=200,
    step=2,
    segmentation=True,
):
    """
    Perform a sweep of epochs for the given VANGAN model and save the resulting images.

    Args:
        args: Command-line arguments.
        vangan_model: A VANGAN object.
        plotter: A GANMonitor-like object.
        test_path (str | Sequence[str]): Directory containing test images *or*
            explicit list/array of filenames to evaluate.
        start (int): Starting epoch number (inclusive).
        end (int): Ending epoch number (inclusive).
        step (int): Epoch stride.
        segmentation (bool): If True, generate segmentation images; otherwise, generate
            fake imaging domain images.
    """
    # ------------------------------------------------------------------ #
    # Robust validation of `test_path`
    # ------------------------------------------------------------------ #
    if test_path is None:
        raise ValueError("`test_path` must not be None.")

    if _is_sequence(test_path):
        if len(test_path) == 0:
            raise ValueError("`test_path` sequence is empty.")
    else:
        # Expecting a directory path at this point
        if str(test_path).strip() == "":
            raise ValueError("`test_path` directory string is empty.")
        if not os.path.isdir(test_path):
            raise ValueError(f"`test_path` directory does not exist: {test_path}")

    # ------------------------------------------------------------------ #
    # Resolve the list of test files
    # ------------------------------------------------------------------ #
    if _is_sequence(test_path):
        # Caller supplied explicit file paths
        testfiles = [os.fspath(p) for p in test_path]
    else:
        # Caller supplied a directory; enumerate its contents
        testfiles = [os.path.join(test_path, f) for f in os.listdir(test_path)]

    # ------------------------------------------------------------------ #
    for epoch in range(start, end + 1, step):
        print(f"\nSampling Epoch {epoch}")
        vangan_model.load_checkpoint(
            epoch=epoch,
            newpath=os.path.join(args.output_dir, "checkpoints"),
        )

        # Create output folder e.g. <output_dir>/Epoch_Sampling/e123
        folder = os.path.join(args.output_dir, "Epoch_Sampling", f"e{epoch}")
        os.makedirs(folder, exist_ok=True)

        filename_prefix = f"e{epoch}_VG_"
        plotter.run_mapping(
            vangan_model,
            testfiles,
            args.INPUT_IMG_SIZE,
            filetext=filename_prefix,
            segmentation=segmentation,
            filepath=folder,
        )