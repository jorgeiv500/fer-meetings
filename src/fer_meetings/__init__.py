import os

# Conservative defaults for constrained local environments where Intel OpenMP
# shared-memory initialization can fail during imports of torch/scikit-learn.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
