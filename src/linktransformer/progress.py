"""Package-wide switch for progress bars.

Set the environment variable LINKTRANSFORMER_PROGRESS=0 (or false/no/off) to turn off every progress bar
linktransformer shows, without passing progress_bar=False to each call. The variable is read on every call,
so setting os.environ["LINKTRANSFORMER_PROGRESS"] = "0" inside a running Python session also works.
"""
import os
import sys
from contextlib import contextmanager

PROGRESS_ENV_VAR = "LINKTRANSFORMER_PROGRESS"
_OFF_VALUES = {"0", "false", "no", "off"}


def progress_enabled(progress_bar: bool = True) -> bool:
    """Whether to show a progress bar: the caller's `progress_bar` flag, unless LINKTRANSFORMER_PROGRESS turns bars off."""
    return bool(progress_bar) and os.getenv(PROGRESS_ENV_VAR, "").strip().lower() not in _OFF_VALUES


@contextmanager
def hf_progress_bars_off_if_disabled():
    """While LINKTRANSFORMER_PROGRESS turns progress bars off, also hide the bars transformers, huggingface_hub and
    datasets show themselves (e.g. "Loading weights" in transformers 5, model downloads, Dataset.map); the previous
    settings are restored. Works as a context manager or, called, as a function decorator."""
    if progress_enabled():
        yield
        return
    from transformers.utils import logging as hf_logging

    # datasets is optional (the train extra) and only matters once something imported it
    datasets = sys.modules.get("datasets")
    transformers_bars = hf_logging.is_progress_bar_enabled()
    datasets_bars = datasets is not None and not datasets.are_progress_bars_disabled()
    hf_logging.disable_progress_bar()
    if datasets_bars:
        datasets.disable_progress_bars()
    try:
        yield
    finally:
        if transformers_bars:
            hf_logging.enable_progress_bar()
        if datasets_bars:
            datasets.enable_progress_bars()
