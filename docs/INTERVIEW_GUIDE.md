# Interview notes

## 30-second project summary

PathoVision is a research-oriented histology image exploration prototype. It reads raster images or a bounded sample of whole-slide tiles, builds an OpenCV tissue mask, measures connected tissue regions, and shows tile-level summaries in Streamlit. It also includes a PyTorch ResNet-18 transfer-learning path for labeled tiles and a PathMNIST pilot. The pipeline demonstrates image processing, pattern recognition, segmentation, region detection, classification, and quantification while documenting where the current heuristics and data do not support clinical claims.

## Walkthrough

1. `pathovision/imaging.py` reads bounded raster previews and computes tissue masks with HSV thresholds and morphology; `pathovision/wsi.py` optionally uses OpenSlide to read a pyramid overview and a bounded sample of level-0 tiles.
2. `pathovision/imaging.py` measures connected components in the tissue mask; `pathovision/pipeline.py` composes the operations and creates the tissue overlay, region boxes, binary mask, and tile-fraction heatmap.
3. `pathovision/train.py` fine-tunes ResNet-18 from ImageNet initialization on either a torchvision `ImageFolder` dataset or the official PathMNIST benchmark, selects a checkpoint using validation macro F1, and can evaluate once on a separate test split. For ImageFolder, an optional CSV audit checks full image coverage and patient/slide group separation before training; it records only per-split group counts.
4. `pathovision/metrics.py` reports a confusion matrix plus per-class precision, recall, F1, accuracy, and macro F1.
5. `app.py` connects uploads, pipeline results, optional checkpoint predictions, and downloads.

## Mapping to medical-image analysis work

- **Image processing / computer vision:** OpenCV color-space thresholding, morphology, connected components, tile extraction, and overlays.
- **Pattern recognition / machine learning:** optional ImageNet-initialized PyTorch ResNet-18 tile classifier; validation selection and held-out PathMNIST evaluation are recorded.
- **Segmentation:** an inspectable HSV/morphology tissue mask, explicitly a baseline rather than learned semantic segmentation.
- **Detection:** bounding boxes for connected tissue-mask regions. These do not identify tumor, lesion, or cell objects.
- **Quantification / quality review:** tile tissue fraction and Laplacian-variance focus proxy, connected-component pixel area and centroid, and WSI physical measurements when valid scale metadata exists. The focus score is scale-dependent and is not an automated accept/reject threshold.
- **High-resolution handling:** OpenSlide region reads and bounded spatial tile sampling avoid loading a whole slide into memory. The current WSI path samples at most 512 candidate tiles, so it does not yet provide exhaustive slide inference or a real-time performance claim.
- **Performance discussion:** the dashboard reports elapsed analysis time for the current run. Compare runs only on the same hardware and similar image/sample settings; this is instrumentation, not a latency benchmark.
- **Traceability / privacy:** each raster/WSI run can export a JSON summary with the settings, image dimensions, slide metadata, measurements, runtime, and model evaluation provenance. The report omits the uploaded filename, local path, and image data.

## Likely questions

**Why tile images?**

Tiles give explicit coordinates and cap model input size. Standard images are capped at a 4,096-pixel longest edge. The optional OpenSlide path reads a pyramid overview and up to 512 level-0 tiles at distributed positions, with coordinates in slide pixels. It is bounded sampled analysis, not exhaustive WSI processing.

The reader was smoke-tested against the public CMU Aperio SVS sample (2,220 × 2,967 pixels): it found 60 overview candidates and produced detailed measurements for 58 level-0 tiles. This checks file reading and the processing path only; it is not an accuracy or segmentation benchmark.

The optional 23,999 × 8,015-pixel Aperio sample has three pyramid levels and is CC BY 4.0. PathoVision uses it to verify that thumbnail-level overview and level-0 region reading work on an actual multilevel slide. Preserve the attribution in `docs/DATA_SOURCES.md`; it has no labels and contributes no classification metrics.

**Why start with a color heuristic?**

It provides a transparent baseline that runs without labels or pretrained pathology weights. It is not robust tissue segmentation, so the next step would compare it against annotated tissue masks across stains and scanners.

**What makes the transfer-learning component real?**

The code loads an ImageNet-initialized ResNet-18, replaces its classification head, fine-tunes it on labeled tiles, saves class names and weights, and loads that checkpoint for tile inference. This project has a completed PathMNIST pilot: it trained on 12,000 patches, selected by macro F1 on a 2,000-patch validation subset, and reached 0.853 accuracy / 0.808 macro F1 on the official 7,180-patch test split. The local checkpoint is excluded from Git, and these benchmark results do not establish performance on high-resolution images or clinical data.

**How should I describe the PathMNIST result?**

Call it a small transfer-learning benchmark pilot, not a diagnostic model or a full-data benchmark result. PathMNIST consists of 28×28 research patches; this run used only a subset of training and validation data. Accuracy and macro F1 differ, so mention class-level metrics and the confusion matrix when discussing errors. The model card records the run configuration, data source, results, and limitations.

**How would you evaluate it credibly?**

Keep patient or slide groups disjoint before extracting tiles, freeze the split, audit the mapping with `--group-manifest`, report class counts and per-class metrics, inspect the confusion matrix, and then test external slides/scanners/stains. The audit validates the supplied manifest against the folders, but it cannot verify that the group codes are truthful. Do not split neighboring tiles from one slide between training and validation.

**What would you improve next?**

Use an OpenSlide-backed streaming reader, validated stain normalization, annotated segmentation data, group-aware training splits, calibration and external validation, and a model-card style record of dataset provenance and intended use.

## Claims to keep accurate

- Say **heuristic tissue masking** for the HSV/morphology baseline.
- Say **completed PathMNIST transfer-learning pilot** when referring to the measured benchmark, and state the subset sizes and held-out test results.
- Never claim clinical use, diagnostic ability, cell detection, or accuracy on medical data without the corresponding validated work and evidence.
