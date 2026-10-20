from io import BytesIO
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pathovision.demo import make_synthetic_demo
from pathovision.imaging import MAX_ANALYSIS_DIMENSION
from pathovision.model_scope import supports_high_resolution_tiles
from pathovision.pipeline import analyze_rgb, encode_png
from pathovision.report import build_analysis_report, encode_analysis_report
from pathovision.wsi import analyze_wsi, is_whole_slide, openslide_available


@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint_path: str, checkpoint_mtime_ns: int):
    from pathovision.classifier import TileClassifier
    return TileClassifier(checkpoint_path)


@st.cache_data(show_spinner=False)
def load_demo_image():
    return make_synthetic_demo()


@st.cache_resource(show_spinner=False)
def load_pathmnist_test_dataset(data_root: str):
    from medmnist import PathMNIST
    return PathMNIST(root=data_root, split="test", download=False)


@st.cache_data(show_spinner=False)
def load_pathmnist_test_sample(data_root: str, sample_index: int):
    dataset = load_pathmnist_test_dataset(data_root)
    image, label = dataset[sample_index]
    class_id = int(np.asarray(label).reshape(-1)[0])
    class_name = dataset.info["label"][str(class_id)]
    return np.asarray(image.convert("RGB")), class_name

st.set_page_config(page_title="PathoVision", page_icon="🔬", layout="wide")
st.title("PathoVision")
st.caption("Research prototype for tile-based histology image exploration")
st.warning("Research use only. This heuristic segmentation and optional experimental classifier are not validated for diagnosis or clinical decisions.")
with st.sidebar:
    st.header("Analysis settings")
    tile_size = st.select_slider("Tile size (pixels; level 0 for WSI)", options=[128, 256, 384, 512], value=256)
    default_checkpoint = "models/pathmnist_resnet18.pt" if (ROOT / "models/pathmnist_resnet18.pt").exists() else "models/resnet18.pt"
    checkpoint = st.text_input("Optional trained model checkpoint", value=default_checkpoint, key="classifier_checkpoint_v2")
    st.caption("Tile scores appear when this points to a compatible, locally trained checkpoint.")
has_pathmnist = (ROOT / "data/medmnist/pathmnist.npz").exists()
source_options = ["Synthetic demo (not medical data)"]
if has_pathmnist:
    source_options.append("PathMNIST held-out test sample")
wsi_sample_path = ROOT / "data/wsi/CMU-1-Small-Region.svs"
if wsi_sample_path.exists() and openslide_available():
    source_options.append("OpenSlide sample slide (CC0)")
multilevel_wsi_path = ROOT / "data/wsi/239551.svs"
if multilevel_wsi_path.exists() and openslide_available():
    source_options.append("Multi-level Aperio sample (CC BY 4.0)")
source_options.append("Upload image")
source_mode = st.radio("Image source", source_options, horizontal=True, key="image_source_v3")
upload = None
if source_mode == "Upload image":
    upload = st.file_uploader("Upload a histology image or whole-slide file", type=[
        "png", "jpg", "jpeg", "tif", "tiff", "svs", "ndpi", "mrxs", "scn", "bif", "czi", "dcm"
    ])
benchmark_truth = None
wsi_path = None
temporary_upload = None
if source_mode == "Synthetic demo (not medical data)":
    input_rgb = load_demo_image()
    output_stem = "synthetic_demo"
    st.info("This generated pattern is for checking the interface and image-processing flow only. It is not a histology image or dataset.")
elif source_mode == "PathMNIST held-out test sample":
    try:
        data_root = str(ROOT / "data/medmnist")
        test_dataset = load_pathmnist_test_dataset(data_root)
        sample_index = st.slider("Held-out test patch index", 0, len(test_dataset) - 1, 0, key="pathmnist_test_index_v1")
        input_rgb, benchmark_truth = load_pathmnist_test_sample(data_root, sample_index)
        output_stem = f"pathmnist_test_sample_{sample_index}"
        st.info(f"Real PathMNIST held-out test patch {sample_index} — ground-truth class: {benchmark_truth}. This is a 28×28 benchmark patch, not a whole-slide image.")
    except Exception as exc:
        st.error(f"Could not load the local PathMNIST test patch: {exc}")
        st.stop()
elif source_mode == "OpenSlide sample slide (CC0)":
    wsi_path = wsi_sample_path
    input_rgb = None
    output_stem = "openslide_sample"
    st.info("Small public-domain Aperio SVS reader sample (2,220 × 2,967 pixels). It exercises the WSI input path but is not a labeled model benchmark.")
elif source_mode == "Multi-level Aperio sample (CC BY 4.0)":
    wsi_path = multilevel_wsi_path
    input_rgb = None
    output_stem = "aperio_multilevel_sample_239551"
    st.info("Public CC BY 4.0 Aperio SVS sample (23,999 × 8,015 pixels; three pyramid levels). Attribution: Whole-slide image '239551', © Toby C. Cornish / WSILabs, licensed under CC BY 4.0. It is an unlabeled reader/processing demo, not model-evaluation data.")
elif upload is not None:
    try:
        output_stem = Path(upload.name).stem
        suffix = Path(upload.name).suffix.lower()
        wsi_suffixes = {".svs", ".ndpi", ".mrxs", ".scn", ".bif", ".czi", ".dcm", ".vms", ".vmu", ".svslide", ".avs"}
        if suffix in wsi_suffixes or suffix in {".tif", ".tiff"}:
            with tempfile.NamedTemporaryFile(prefix="pathovision_", suffix=suffix, delete=False) as temp_file:
                temp_file.write(upload.getvalue())
                temporary_upload = Path(temp_file.name)
            if suffix in wsi_suffixes or is_whole_slide(temporary_upload):
                wsi_path = temporary_upload
                input_rgb = None
            else:
                temporary_upload.unlink(missing_ok=True)
                temporary_upload = None
        if wsi_path is None:
            decoded = Image.open(BytesIO(upload.getvalue()))
            original_width, original_height = decoded.size
            decoded.thumbnail((MAX_ANALYSIS_DIMENSION, MAX_ANALYSIS_DIMENSION))
            input_rgb = np.asarray(decoded.convert("RGB"))
            if input_rgb.shape[1] != original_width or input_rgb.shape[0] != original_height:
                st.warning(
                    f"Image downsampled from {original_width:,} × {original_height:,} to "
                    f"{input_rgb.shape[1]:,} × {input_rgb.shape[0]:,} for bounded-memory preview analysis. "
                    "Tile sizes and coordinates refer to the resized analysis image; this is not whole-slide streaming."
                )
    except Exception as exc:
        if temporary_upload is not None:
            temporary_upload.unlink(missing_ok=True)
        st.error(f"Could not read the uploaded image: {exc}")
        st.stop()
else:
    st.info("Choose a demo source or upload an image to begin.")
    st.stop()
classifier = None
model_path = ROOT / checkpoint
if model_path.exists():
    try:
        classifier = load_classifier(str(model_path.resolve()), model_path.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Could not load classifier checkpoint: {exc}")
with st.spinner("Analyzing image tiles..."):
    # The synthetic UI demo is not medical data; PathMNIST inference uses only its known benchmark patch.
    inference_classifier = classifier if upload is not None or wsi_path is not None else None
    if (inference_classifier is not None
            and not supports_high_resolution_tiles(classifier.training_config)):
        st.info(
            "The PathMNIST classifier is trained on 28×28 benchmark patches. "
            "It is reserved for the held-out PathMNIST sample demo and is not applied to uploaded images or whole-slide tiles."
        )
        inference_classifier = None
    if wsi_path is not None:
        try:
            result = analyze_wsi(wsi_path, tile_size, inference_classifier)
            input_rgb = result["thumbnail"]
        except Exception as exc:
            st.error(f"Could not analyze the whole-slide image: {exc}")
            st.stop()
        finally:
            if temporary_upload is not None:
                temporary_upload.unlink(missing_ok=True)
        metadata = result.get("slide_metadata")
        if isinstance(metadata, dict):
            metadata_parts = [f"Slide format: {metadata.get('vendor') or 'unknown'}"]
            if metadata.get("objective_power"):
                metadata_parts.append(f"objective: {metadata['objective_power']}×")
            if metadata.get("mpp_x") is not None and metadata.get("mpp_y") is not None:
                metadata_parts.append(f"scale: {metadata['mpp_x']:.4f} × {metadata['mpp_y']:.4f} µm/px")
            metadata_parts.append(f"pyramid levels: {metadata.get('pyramid_levels', 'unknown')}")
            st.caption(" · ".join(metadata_parts) + ". Tile CSV includes physical coordinates where slide metadata provides pixel scale.")
        else:
            st.warning("Slide analysis completed, but the reader did not return slide metadata for this file.")
    else:
        result = analyze_rgb(input_rgb, tile_size, inference_classifier)
benchmark_prediction = None
if benchmark_truth is not None and classifier is not None:
    if benchmark_truth in classifier.classes:
        height, width = input_rgb.shape[:2]
        prediction = classifier.predict_tiles(input_rgb, [{"x": 0, "y": 0, "width": width, "height": height}])
        benchmark_prediction = prediction["tiles"][0]
    else:
        st.warning("The selected checkpoint does not use PathMNIST class labels, so this patch cannot be scored with it.")
if benchmark_prediction is not None:
    predicted = benchmark_prediction["label"]
    x, y, z = st.columns(3)
    x.metric("Known test label", benchmark_truth)
    y.metric("Model prediction", predicted)
    z.metric("Match", "Yes" if predicted == benchmark_truth else "No")
    st.caption(f"Model confidence: {benchmark_prediction['confidence']:.3f}. Confidence is not calibrated diagnostic certainty.")
elif benchmark_truth is not None:
    st.info("A compatible trained PathMNIST checkpoint is needed to show a prediction for this patch.")
if benchmark_truth is None:
    a, b, c, d = st.columns(4)
    a.metric("Image dimensions", f"{result['width']:,} × {result['height']:,}")
    b.metric("Tissue area (heuristic)", f"{100*result['tissue_area_fraction']:.1f}%")
    c.metric("Analyzed tissue tiles", f"{result['tile_count']:,}")
    d.metric("Analysis runtime", f"{result['analysis_time_seconds']:.2f} s")
    st.caption(f"Connected tissue regions measured: {len(result.get('regions', [])):,}. Boxes follow the heuristic mask; they do not identify tumors, lesions, or cells.")
    if result.get("candidate_tile_count") is not None:
        st.caption(
            f"Overview mask identified {result['candidate_tile_count']:,} candidate tiles; "
            f"detailed level-0 analysis sampled {result['tile_count']:,}. Coordinates are level-0 pixels."
        )
    left, center, right = st.columns(3)
    input_caption = "Whole-slide overview thumbnail" if wsi_path is not None else (
        "Input image" if upload else ("PathMNIST held-out test sample" if benchmark_truth is not None else "Synthetic interface demo (not medical data)")
    )
    left.image(input_rgb, caption=input_caption, width="stretch")
    center.image(result["overlay"], caption="Heuristic tissue overlay + tile grid + connected-region boxes", width="stretch")
    heatmap_caption = "Sampled level-0 tile tissue fraction on overview" if wsi_path is not None else "Tile tissue fraction (low → blue, high → red)"
    right.image(result["tile_heatmap"], caption=heatmap_caption, width="stretch")
    st.subheader("Tile measurements")
    if result["tiles"]:
        frame = pd.DataFrame(result["tiles"])
        st.dataframe(frame, width="stretch", hide_index=True)
        chart_data = frame.assign(tile=[f"({x},{y})" for x, y in zip(frame["x"], frame["y"])])
        tissue_chart, focus_chart = st.columns(2)
        tissue_chart.bar_chart(chart_data, x="tile", y="tissue_fraction", width="stretch")
        focus_chart.bar_chart(chart_data, x="tile", y="focus_laplacian_variance", width="stretch")
        st.caption("Focus proxy is Laplacian variance. It depends on scale and tissue content; compare tiles within the same image and do not treat it as a universal pass/fail threshold.")
        st.download_button("Download tile measurements CSV", frame.to_csv(index=False).encode("utf-8"),
                           file_name=f"{output_stem}_tiles.csv", mime="text/csv")
    else:
        st.info("No tiles passed the tissue threshold. Try a different image or tile size.")
    if result.get("regions"):
        st.subheader("Connected tissue-region measurements")
        region_frame = pd.DataFrame(result["regions"])
        st.dataframe(region_frame, width="stretch", hide_index=True)
        st.download_button("Download region measurements CSV", region_frame.to_csv(index=False).encode("utf-8"),
                           file_name=f"{output_stem}_regions.csv", mime="text/csv")
else:
    st.image(input_rgb, caption="PathMNIST held-out test sample", width=224)
    st.caption("This 28×28 benchmark patch is shown for classifier evaluation; tissue-segmentation overlays are intended for larger images.")
if result["classifier"]:
    st.subheader("Experimental tile classifier")
    st.caption("Scores are model confidence values, not calibrated probabilities or diagnostic certainty.")
    st.dataframe(pd.DataFrame(result["classifier"]["tiles"]), width="stretch", hide_index=True)

if classifier is not None:
    val = classifier.validation_metrics
    test_metrics = classifier.test_metrics
    if val or test_metrics:
        dataset_name = classifier.training_config.get("dataset", "custom image-folder data")
        dataset_title = "PathMNIST" if dataset_name == "pathmnist" else dataset_name
        st.subheader(f"Model evaluation — {dataset_title}")
        if dataset_name == "pathmnist":
            train_n = classifier.training_config.get("train_samples")
            val_n = classifier.training_config.get("validation_samples")
            test_n = classifier.training_config.get("test_samples")
            st.caption(
                f"PathMNIST 28×28 research benchmark. Training used {train_n:,} train and {val_n:,} validation patches; "
                f"held-out test evaluation used {test_n:,} patches. These results do not establish performance on "
                "whole-slide or high-resolution images and are not clinical evidence."
            )
        else:
            st.caption("Checkpoint-reported metrics. Reliability depends on the dataset provenance and split design.")
        if val:
            x, y = st.columns(2)
            x.metric("Validation accuracy", f"{val['accuracy']:.3f}")
            y.metric("Validation macro F1", f"{val['macro_f1']:.3f}")
            metric_frame = pd.DataFrame(val["per_class"])
            st.caption("Per-class validation scores")
            st.dataframe(metric_frame, width="stretch", hide_index=True)
            metric_index = metric_frame.set_index("class")
            score_columns = st.columns(3)
            for column, metric in zip(score_columns, ("precision", "recall", "f1")):
                with column:
                    st.bar_chart(metric_index[[metric]], y_label=metric.title())
            st.caption("Validation confusion matrix — rows are actual classes; columns are predicted classes.")
            class_names = val.get("classes") or [row["class"] for row in val["per_class"]]
            confusion = pd.DataFrame(val["confusion_matrix"], index=class_names, columns=class_names)
            confusion.index.name = "Actual ↓ / Predicted →"
            st.dataframe(confusion, width="stretch")
        if test_metrics:
            x, y = st.columns(2)
            x.metric("Held-out test accuracy", f"{test_metrics['accuracy']:.3f}")
            y.metric("Held-out test macro F1", f"{test_metrics['macro_f1']:.3f}")
            test_frame = pd.DataFrame(test_metrics["per_class"])
            st.caption("Per-class held-out test scores")
            st.dataframe(test_frame, width="stretch", hide_index=True)
            test_index = test_frame.set_index("class")
            test_score_columns = st.columns(3)
            for column, metric in zip(test_score_columns, ("precision", "recall", "f1")):
                with column:
                    st.bar_chart(test_index[[metric]], y_label=metric.title())
            st.caption("Held-out test confusion matrix — rows are actual classes; columns are predicted classes.")
            test_class_names = test_metrics.get("classes") or [row["class"] for row in test_metrics["per_class"]]
            test_confusion = pd.DataFrame(
                test_metrics["confusion_matrix"], index=test_class_names, columns=test_class_names
            )
            test_confusion.index.name = "Actual ↓ / Predicted →"
            st.dataframe(test_confusion, width="stretch")
if benchmark_truth is None:
    run_report = build_analysis_report(
        result,
        source=source_mode,
        tile_size=tile_size,
        classifier=classifier,
    )
    st.download_button("Download run report JSON", encode_analysis_report(run_report),
                       file_name="pathovision_run_report.json", mime="application/json")
    st.download_button("Download overlay PNG", encode_png(result["overlay"]),
                       file_name=f"{output_stem}_overlay.png", mime="image/png")
    st.download_button("Download tissue mask PNG", encode_png(result["mask"]),
                       file_name=f"{output_stem}_mask.png", mime="image/png")
    st.download_button("Download tile heatmap PNG", encode_png(result["tile_heatmap"]),
                       file_name=f"{output_stem}_tile_heatmap.png", mime="image/png")
