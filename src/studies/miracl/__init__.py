__version__ = "0.1.0"
__author__ = "RORetrieval Team"

# Relative imports so this works as a real package (`src.studies.miracl`).
# The submodules themselves fall back to flat imports when the miracl/
# directory is on sys.path directly (how run.sh invokes them).
from .miracl_downloader import MIRACLDownloader
from .miracl_preprocessor import MIRACLPreprocessor
from .miracl_pipeline import MIRACLPipeline

__all__ = [
    "MIRACLDownloader",
    "MIRACLPreprocessor",
    "MIRACLPipeline"
]
