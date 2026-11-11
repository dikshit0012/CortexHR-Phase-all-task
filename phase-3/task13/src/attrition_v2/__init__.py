"""CortexHR Task 13: Attrition Model V2 (gradient boosting vs the honest baselines).

Reuses the Task 2 feature pipeline and the Task 3 fold builder from the ``cortexhr`` package
(``task 4/files/cortexhr-ml/src``). Nothing from Tasks 1-12 is copied or modified.
"""
from .config import MODEL_VERSION  # noqa: F401  (importing config also puts ``cortexhr`` on sys.path)

__all__ = ["MODEL_VERSION"]
