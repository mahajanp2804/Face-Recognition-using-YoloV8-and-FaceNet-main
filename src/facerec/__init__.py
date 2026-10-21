"""Face recognition attendance system: YOLOv8 detection + MTCNN alignment + FaceNet embeddings.

Submodules that need torch / ultralytics (``detector``, ``embedder``, ``enroll``,
``recognize``) are not imported here, so ``matcher``, ``attendance`` and ``config``
stay importable with only numpy and PyYAML installed.
"""

__version__ = "0.1.0"
