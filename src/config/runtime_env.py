"""Host-specific env defaults before loading PyTorch + FAISS (avoids native crashes)."""

from __future__ import annotations

import os
import platform


def apply_darwin_openmp_mitigations() -> None:
    """On macOS, limit BLAS/OpenMP threading and allow duplicate OpenMP runtimes.

    Without this, combining SentenceTransformer (PyTorch/MPS) and faiss-cpu often
    ends in a segmentation fault right after ``import faiss`` or on first ``add``/``search``.
    Uses ``setdefault`` so values you export in the shell still win.
    """
    if platform.system() != "Darwin":
        return
    for key, value in (
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ):
        os.environ.setdefault(key, value)
