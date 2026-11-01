# Library compliance review

Authority: assignment section 2a. The runtime dependency allowlist is Python stdlib, PyTorch, NumPy, OpenCV and Matplotlib. All model layers are explicitly assembled; there are no model-zoo imports, pretrained downloads, alternate engines, library metrics, morphology, clustering or max-flow operations. The selected track does not require implementing Track A.

| Import group | Purpose | Permitted functionality |
|---|---|---|
| torch; torch.nn; torch.nn.functional | Original convolutional networks, BCE, explicit Dice arithmetic, boundary tensor slices and pooling reference tests | tensors, autograd, nn layers/functions, optimizers |
| torch.utils.data | Dataset, zero-worker DataLoader | authorized data loading |
| numpy | uint8 image arrays, validation, conversion, plot overlays, deterministic diagnostic image, 25 whole-array OR passes for exact boundary windows | image/tensor array operations and plotting inputs; no library segmentation/metrics |
| cv2 | imread, imwrite, resize, cvtColor, thread configuration/query | file I/O, resizing, color conversion and required runtime setup |
| matplotlib; matplotlib.pyplot; matplotlib.widgets.PolygonSelector | Measured learning curves, comparisons, mask visualizations, manual annotation interactions | plotting; annotation is expressly allowed with any tool |
| pathlib, os, shutil, tempfile, uuid | File handling, artifact paths, test fixtures, actual saved size | standard library |
| json, csv | Configs, manifests/provenance/results serialization | standard library |
| argparse, sys | CLI arguments, stderr, return codes | standard library |
| random | Seeded deterministic train/val split | standard library |
| math | Finite/range checks | standard library |
| time, datetime | Official process_time, supplemental train elapsed time, timestamps | standard library |
| hashlib | SHA-256 artifact identity | standard library |
| platform, winreg | Actual software/CPU metadata (read-only OS registry) | standard library |
| urllib.request, zipfile | Official dataset download/extraction only | standard library I/O; archive extraction is not weight compression |
| ast | Source audit | standard library |
| unittest, unittest.mock, contextlib, io | Analytic tests and controlled fixtures | standard library |
| subprocess | Fresh-process integration tests and optional pdflatex build only | standard library; never imported/called by the inference pipeline |
| src.*, scripts.*, tests.*, evaluate | Local original implementation and CLI forwarding | repository source, not third-party packages |

The complete per-file, per-line import inventory is generated in `results/source_audit.json`; regenerate with `python -m scripts.audit`. The audit scans actual AST imports and direct cv2 calls, supplemented by manual review of operations. It is a guardrail, not proof against arbitrary disguised dynamic calls. No dynamic loading or disguised calls are used.

The analysis environment's PyMuPDF was used **only to extract/read/render PDFs**, and is not imported by project source, requirements or prediction. No PDF-processing package is needed to train or infer. LaTeX compilation (pdflatex or portable Tectonic) is an external document-build step, separate from the segmentation runtime. Its subprocess is not used for timing. Portable compiler binaries and caches stay under ignored `tmp/` and are excluded from packaging.

No uncertain third-party segmentation functionality is included. The project follows the prompt's permitted tensor-based brightness/contrast and flips. Annotation support is a manual plotting UI plus a custom torch ray-crossing rasterizer, not an automated segmentation library.
