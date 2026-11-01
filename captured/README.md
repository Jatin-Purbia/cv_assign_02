# Genuine phone captures — pending

Required literal folders, verified from the rendered PDF: `original/`, `images_256/`, `masks_256/`. Text extraction lost the underscore glyphs. No authentic photos or annotations were supplied at project initialization. Do not use generated/downloaded replacements or train on these photos.

Take 5–10 photos around college/hostel, each with a clear everyday object. Cover at least three categories: both indoor/outdoor, both daylight/tube light, both near/far, clutter, partial occlusion. For a "both" category, include and document examples of each alternative. Choose examples before evaluating the model; do not discard failures.

Use the same stem for original JPG and resized/mask PNG. Directly resize photographs to 256x256 with INTER_AREA. Draw masks **on the resized image**, with object 255 and background 0; no anti-aliasing. `python -m scripts.annotate --image path.jpg` provides a single-polygon Matplotlib editor for simple objects. Finish the polygon then press Enter to save; close to cancel. For multiple disconnected parts/holes, use a manual editor such as Paint/GIMP and export binary PNG. No library segmentation or morphology is used.

Create `captured/metadata.json` after real collection, using this schema (replace example text and IDs; false attestations must never be set true):

```json
{
  "authentic_phone_photos": false,
  "manually_annotated": false,
  "capture_process": "Describe your phone, location and date",
  "photos": [
    {
      "id": "actual_file_stem",
      "object": "actual photographed object",
      "annotation_method": "manual polygon on resized image",
      "conditions": ["indoor_outdoor", "clutter", "occlusion"],
      "scene": "indoor or outdoor",
      "lighting": "daylight or tube light",
      "distance": "near or far",
      "notes": "Explain what is visible and how conditions are covered"
    }
  ]
}
```

The validator checks 5–10 paired stems, PNGs, dimensions, binary values, per-photo documentation, attestation and >=3 category labels. **A person must verify authenticity and semantic coverage**, especially both-alternative categories. Evaluate only after final model freeze. Generate all overlays, including failures. Do not use any captured metric for tuning.
