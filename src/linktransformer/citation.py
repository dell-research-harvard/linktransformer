from __future__ import annotations

from typing import Literal


APA_CITATION = (
    "If you use LinkTransformer in your research, please cite the following paper:\n\n"
    "Arora, A., & Dell, M. (2024). "
    "LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models "
    "[Conference paper]. https://doi.org/10.18653/v1/2024.acl-demos.21"
)


def citation(format: Literal["apa"] = "apa") -> str:
    """Return the package citation string.

    Parameters
    ----------
    format:
        Currently only supports ``'apa'``.
    """

    if format != "apa":
        raise ValueError("Only APA citation format is supported: format='apa'")
    return APA_CITATION


def print_citation() -> None:
    """Print the APA citation string to stdout."""

    print(APA_CITATION)
