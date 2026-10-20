# Resume and interview brief

Use the measured PathMNIST pilot only with its dataset and subset context. The WSI sample validates the reader and visualization path, not classification accuracy.

## Resume bullets

- Built **PathoVision**, a Python application for histology image exploration, combining OpenCV tissue-mask heuristics, connected-region measurements, tile-level tissue/focus summaries, and a Streamlit review dashboard.
- Implemented bounded whole-slide image reading with OpenSlide: pyramid overview plus up to 512 distributed level-0 tiles, with pixel coordinates and physical measurements when valid microns-per-pixel metadata is available.
- Fine-tuned an ImageNet-initialized PyTorch ResNet-18 on a reproducible PathMNIST pilot (12,000 train / 2,000 validation patches); achieved **0.853 accuracy and 0.808 macro F1** on the official 7,180-patch test split after validation-based checkpoint selection.
- Added per-class precision/recall/F1, confusion-matrix reporting, patient/slide split-manifest auditing for custom datasets, bounded-batch inference, privacy-conscious run reports, and automated tests.

**Compact version:** Built a Python, OpenCV, PyTorch, and Streamlit histology-imaging prototype with bounded OpenSlide tile sampling, transparent tissue-mask heuristics, and ResNet-18 transfer learning; evaluated a PathMNIST pilot at 0.853 test accuracy / 0.808 macro F1 and documented data, split, and clinical-use limitations.

Only include the measured scores if prepared to explain that PathMNIST uses 28×28 research patches and that the pilot used subsets of train and validation data. Do not imply that those scores measure WSI, patient-level, or clinical performance.

## 30-second project pitch

“PathoVision is a local histology image-analysis prototype focused on the engineering problems around large images. It reads raster images and, optionally, pyramid whole-slide files; samples a bounded number of tiles; computes an inspectable OpenCV tissue-mask baseline and region/tile measurements; and presents the results in Streamlit. I also implemented a real PyTorch ResNet-18 fine-tuning and evaluation path. On a documented PathMNIST subset-training pilot it achieved 0.853 accuracy and 0.808 macro F1 on the official test split. I keep the 28×28 benchmark classifier separate from WSI inference, and the project makes no diagnostic or clinical-performance claim.”

## Role-relevant evidence

| Role area | Evidence to discuss | Boundary to state clearly |
| --- | --- | --- |
| Image processing and computer vision | HSV thresholding, morphology, connected components, overlays, tile extraction | The tissue mask is a transparent heuristic, not learned semantic segmentation |
| High-resolution image handling | OpenSlide pyramid metadata, region reads, bounded spatial sampling, pixel/physical coordinates | Detailed analysis samples at most 512 level-0 tiles; it is not exhaustive or a real-time benchmark |
| Classification and deep learning | ImageNet-initialized ResNet-18, replaced head, fine-tuning, checkpointing, batched inference | PathMNIST weights are used only with the held-out 28×28 benchmark-patch demo |
| Evaluation and data discipline | Validation macro-F1 checkpoint selection, held-out test pass, confusion matrix, class metrics, group-manifest audit | The manifest audit checks supplied mappings; it cannot prove group identifiers are truthful |
| Software engineering | Modular Python package, Streamlit application, run-report export, tests, dependency setup | Portfolio/research prototype, not a deployed clinical service |

## Interview questions

### Why tile a whole-slide image?

Whole-slide images can exceed memory if decoded as one array, while a model typically accepts fixed-size inputs. A pyramid overview helps locate candidate tissue; bounded region reads cap memory and provide tile coordinates. PathoVision samples at most 512 level-0 tiles per run. This is a practical reader/processing demonstration, not exhaustive slide inference.

### Why use an HSV heuristic before a learned segmentation model?

It is fast, inspectable, needs no segmentation labels, and gives a baseline for identifying tissue-positive regions. Its thresholds can fail when stain, scanner, illumination, or artifacts change. A credible next step is to compare against annotated masks across slides and report Dice/IoU by source, rather than presenting the heuristic as validated segmentation.

### What does the transfer-learning result show?

The training code starts from ImageNet ResNet-18 weights, replaces the final classification layer, fine-tunes on labeled data, selects the checkpoint using validation macro F1, and calculates metrics once on the held-out test split. The pilot used 12,000 of 89,996 training patches and 2,000 of 10,004 validation patches, then all 7,180 test patches. It reached 0.853 test accuracy and 0.808 macro F1. It demonstrates an end-to-end training/evaluation workflow on a small-patch public benchmark; it does not demonstrate high-resolution WSI or clinical performance.

### Why report macro F1 as well as accuracy?

Accuracy weights every example equally, so frequent classes can dominate it. Macro F1 averages class-level F1 equally and can expose weaker performance on less frequent classes. The difference in this pilot (0.853 accuracy versus 0.808 macro F1) is a reason to inspect per-class metrics and the confusion matrix, not a reason to claim a particular clinical failure mode.

### How do you reduce tile leakage?

Split by patient or slide before making tiles, then keep all tiles from one group in a single split. For custom ImageFolder training, the optional CSV audit checks that every loaded image is represented once and that a supplied group does not cross splits. It records split group counts rather than identifiers. The audit cannot establish that the user-provided groups correspond to real patients/slides, so provenance still needs review.

### What would you do next for a stronger medical-imaging evaluation?

Use a documented patient/slide-disjoint cohort, annotated tissue masks, and an external scanner or institution; report class balance, per-class metrics, calibration, and uncertainty; then measure memory and latency under fixed hardware and sampling settings. Any clinical intended use would require appropriate expert, regulatory, and prospective validation beyond this portfolio project.

## Demo sequence

1. Show the generated interface sample and state explicitly that it is synthetic.
2. Open the public multilevel Aperio reader sample and explain overview versus bounded tile reads.
3. Point out that the overlay and region boxes come from the heuristic mask, then inspect tile and physical-coordinate summaries where scale metadata is available.
4. Show the held-out PathMNIST patch demo and compare label with prediction; identify it as 28×28 benchmark data.
5. Show the model card, data attribution, tests, and the explicit boundary between the benchmark classifier and WSI processing.

## Project links

- [README and setup](../README.md)
- [Architecture](ARCHITECTURE.md)
- [Interview walkthrough](INTERVIEW_GUIDE.md)
- [PathMNIST pilot model card](../MODEL_CARD.md)
- [Dataset sources and attribution](DATA_SOURCES.md)
