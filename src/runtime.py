"""Import this module before importing numerical libraries in every entry point."""
import os

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS')
for variable in THREAD_VARIABLES:
    os.environ[variable] = '1'


def configure() -> None:
    """Restrict OpenCV and PyTorch CPU execution to one thread."""
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    # No inter-op worker parallelism is needed by this eager model.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
