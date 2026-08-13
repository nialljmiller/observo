#!/usr/bin/env python3
"""Incremental wildlife detection for the weather-station camera.

MegaDetector first localizes any animal, rather than pretending a COCO model
knows classes it was never trained on.  ResNet then supplies a best-effort
species label for the animal crop.  Results are written as annotated JPEGs and
a JSON manifest that browsers can consume without directory listing enabled.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "md_v5a.0.1.pt"
LABELS_FILE = BASE_DIR / "imagenet_classes.txt"
DEFAULT_CLASSES = [
    "bird", "deer", "squirrel", "rabbit", "cat", "dog", "fox", "raccoon",
    "opossum", "skunk", "coyote", "rodent", "groundhog", "weasel", "badger",
    "turkey", "hawk", "owl", "crow", "woodpecker", "animal",
]

CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageNet does not have a literal white-tailed-deer class.  It generally maps
# deer photographs to visually related ungulates, which we normalize here.
SPECIES_ALIASES = {
    "impala": "deer", "gazelle": "deer", "hartebeest": "deer",
    "ibex": "deer", "bighorn": "deer", "ram": "deer",
    "fox squirrel": "squirrel", "marmot": "groundhog",
    "wood rabbit": "rabbit", "hare": "rabbit", "polecat": "skunk",
    "red fox": "fox", "grey fox": "fox", "kit fox": "fox",
    "tabby": "cat", "tiger cat": "cat", "Egyptian cat": "cat",
    "cougar": "cat", "lynx": "cat", "timber wolf": "coyote",
    "red wolf": "coyote", "white wolf": "coyote",
}

KNOWN_ANIMALS = {
    "bird", "deer", "squirrel", "rabbit", "cat", "dog", "fox", "raccoon",
    "opossum", "skunk", "coyote", "rodent", "groundhog", "weasel", "badger",
    "turkey", "hawk", "owl", "crow", "raven", "woodpecker", "mouse", "rat",
    "vole", "chipmunk", "mink", "ferret", "otter", "beaver", "porcupine",
    "armadillo", "bear", "hog", "boar", "bird", "robin", "jay", "finch",
    "hummingbird", "goose", "duck", "heron", "egret", "vulture", "falcon",
}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path, default):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def load_detector(model_file: Path = MODEL_FILE):
    if not model_file.exists():
        raise FileNotFoundError(
            f"Wildlife model not found at {model_file}; install MegaDetector v5a weights"
        )
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=str(model_file),
        skip_validation=True, verbose=False,
    )
    model.classes = [0]  # MegaDetector: 0 animal, 1 person, 2 vehicle
    model.max_det = 20
    return model


def load_classifier():
    # V1 performs more reliably on this camera's IR/green night imagery than
    # the newer V2 preprocessing (validated against the station's deer frame).
    classifier = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    classifier.eval()
    return classifier


def load_labels() -> list[str]:
    with LABELS_FILE.open(encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip()]


def _normalized_species(label: str, allowed: set[str]) -> str | None:
    lower = label.lower()
    normalized = SPECIES_ALIASES.get(lower, lower)
    if normalized in allowed:
        return normalized
    for animal in allowed - {"animal"}:
        if re.search(rf"\b{re.escape(animal)}\b", lower):
            return animal
    if normalized in KNOWN_ANIMALS:
        return normalized
    return None


def classify_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    classifier,
    labels: list[str],
    allowed: set[str],
) -> tuple[str, float]:
    crop = image.crop(box)
    if crop.width < 8 or crop.height < 8:
        return "animal", 0.0
    tensor = CLASSIFIER_TRANSFORM(crop).unsqueeze(0)
    with torch.inference_mode():
        probabilities = F.softmax(classifier(tensor), dim=1)[0]
    top_probabilities, top_indices = probabilities.topk(10)
    for probability, index in zip(top_probabilities, top_indices):
        species = _normalized_species(labels[index.item()], allowed)
        if species:
            return species, float(probability.item())
    return "animal", float(top_probabilities[0].item())


def detect_animals(image: Image.Image, detector, threshold: float) -> pd.DataFrame:
    detector.conf = threshold
    with torch.inference_mode():
        result = detector(image, size=1280)
    detections = result.pandas().xyxy[0]
    if detections.empty:
        return detections
    return detections[
        (detections["name"] == "animal") &
        (detections["confidence"] >= threshold)
    ].copy()


def detect_animal_batch(images: list[Image.Image], detector, threshold: float) -> list[pd.DataFrame]:
    """Run frames together to avoid paying model overhead for every image."""
    detector.conf = threshold
    with torch.inference_mode():
        result = detector(images, size=1280)
    filtered = []
    for detections in result.pandas().xyxy:
        filtered.append(detections[
            (detections["name"] == "animal") &
            (detections["confidence"] >= threshold)
        ].copy())
    return filtered


def _font(size: int = 44):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(image: Image.Image, detections: list[dict]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = _font(max(20, round(output.width / 70)))
    line_width = max(3, round(output.width / 700))
    for detection in detections:
        box = detection["box"]
        species = detection["species"].replace("_", " ").title()
        label = f"{species}  {detection['detector_confidence']:.0%}"
        draw.rectangle(box, outline="#ff3b30", width=line_width)
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        text_height = text_box[3] - text_box[1] + 8
        text_y = max(0, box[1] - text_height)
        draw.rectangle((box[0], text_y, text_box[2] + 8, text_y + text_height), fill="#ff3b30")
        draw.text((box[0] + 4, text_y + 2), label, fill="white", font=font)
    return output


def _timestamp_for(path: str) -> datetime | None:
    try:
        return datetime.strptime(Path(path).stem, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _write_detection_csv(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    fields = [
        "image", "timestamp", "species", "detector_confidence",
        "species_confidence", "xmin", "ymin", "xmax", "ymax",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for detection in record["detections"]:
                box = detection["box"]
                writer.writerow({
                    "image": record["filename"],
                    "timestamp": record["timestamp"],
                    "species": detection["species"],
                    "detector_confidence": detection["detector_confidence"],
                    "species_confidence": detection["species_confidence"],
                    "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3],
                })
    os.replace(temporary, path)


def publish_outputs(output_dir: Path, records: list[dict]) -> None:
    records.sort(key=lambda record: record["timestamp"], reverse=True)
    _atomic_json(output_dir / "detections.json", records)
    _write_detection_csv(output_dir / "animal_detections.csv", records)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(records),
        "images": records,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    if not records:
        return

    latest = records[0]
    latest_source = output_dir / latest["filename"]
    latest_image = output_dir / "latest_animal.jpg"
    temporary_image = output_dir / "latest_animal.tmp.jpg"
    shutil.copy2(latest_source, temporary_image)
    os.replace(temporary_image, latest_image)
    summary = dict(latest)
    summary["image_url"] = "/bigdata/weather_station/images/birds/latest_animal.jpg"
    _atomic_json(output_dir / "latest_detection.json", summary)


def include_legacy_gallery_images(output_dir: Path, records: list[dict]) -> None:
    """Keep annotated images produced by the former detector in the gallery."""
    recorded = {record.get("filename") for record in records}
    excluded = {"latest_animal.jpg"}
    for path in output_dir.glob("*.jpg"):
        if path.name in recorded or path.name in excluded:
            continue
        timestamp = _timestamp_for(path.name)
        if timestamp is None:
            continue
        records.append({
            "filename": path.name,
            "timestamp": timestamp.isoformat(),
            "detections": [{
                "species": "animal",
                "detector_confidence": 0.0,
                "species_confidence": 0.0,
                "box": [0, 0, 0, 0],
                "legacy": True,
            }],
        })


def run_detection_pipeline(
    image_dir,
    output_dir,
    confidence_threshold=0.2,
    log_file="processed_images.json",
    bot=None,
    hours_back=3,
    target_classes=None,
):
    """Detect newly arrived wildlife images and refresh web-facing metadata."""
    del bot  # Posting is deliberately separate from the detection pipeline.
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file)
    allowed = {item.lower() for item in (target_classes or DEFAULT_CLASSES)}
    allowed.add("animal")

    processed_value = _load_json(log_path, [])
    if isinstance(processed_value, dict):
        processed = set(processed_value.get("processed", []))
    else:  # Migrate the old list-only format.
        processed = set(processed_value)
    records = _load_json(output_dir / "detections.json", [])
    if not isinstance(records, list):
        records = []
    include_legacy_gallery_images(output_dir, records)
    recorded_files = {record.get("filename") for record in records}

    candidates = []
    for filename in sorted(glob.glob(str(image_dir / "*.jpg"))):
        if Path(filename).name == "latest.jpg":
            continue
        timestamp = _timestamp_for(filename)
        if timestamp:
            candidates.append((filename, timestamp))
    if not candidates:
        publish_outputs(output_dir, records)
        return

    newest = candidates[-1][1]
    cutoff = newest - timedelta(hours=hours_back)
    pending = [
        item for item in candidates
        if item[1] >= cutoff and Path(item[0]).name not in processed
    ]
    if not pending:
        publish_outputs(output_dir, records)
        logging.info("Wildlife detection: no new images to process")
        return

    logging.info("Wildlife detection: processing %d new images", len(pending))
    detector = load_detector()
    classifier = None
    labels = None

    batch_size = 4
    for offset in range(0, len(pending), batch_size):
        batch_items = pending[offset:offset + batch_size]
        loaded = []
        for filename, timestamp in batch_items:
            try:
                with Image.open(filename) as source:
                    # Camera images arrive at the server in their final display
                    # orientation.  Do not retain the former 180° correction.
                    image = source.convert("RGB")
                loaded.append((filename, timestamp, image))
            except Exception as exc:
                logging.warning("Wildlife detection skipped %s: %s", Path(filename).name, exc)
        if not loaded:
            continue
        try:
            frames = detect_animal_batch(
                [item[2] for item in loaded], detector, confidence_threshold,
            )
        except Exception as exc:
            logging.warning("Wildlife detection batch failed: %s", exc)
            continue

        for (filename, timestamp, image), frame in zip(loaded, frames):
            basename = Path(filename).name
            if not frame.empty:
                if classifier is None:
                    classifier = load_classifier()
                    labels = load_labels()
                detections = []
                for row in frame.itertuples(index=False):
                    box = tuple(round(value) for value in (row.xmin, row.ymin, row.xmax, row.ymax))
                    species, species_confidence = classify_crop(image, box, classifier, labels, allowed)
                    detections.append({
                        "species": species,
                        "detector_confidence": round(float(row.confidence), 5),
                        "species_confidence": round(species_confidence, 5),
                        "box": list(box),
                    })
                annotated = annotate(image, detections)
                temporary = output_dir / f".{basename}.tmp"
                annotated.save(temporary, format="JPEG", quality=92)
                os.replace(temporary, output_dir / basename)
                if basename not in recorded_files:
                    records.append({
                        "filename": basename,
                        "timestamp": timestamp.isoformat(),
                        "detections": detections,
                    })
                    recorded_files.add(basename)
                logging.info("Wildlife detected in %s: %s", basename, ", ".join(d["species"] for d in detections))
            processed.add(basename)
        _atomic_json(log_path, {"processed": sorted(processed)})

    publish_outputs(output_dir, records)
    logging.info("Wildlife detection complete: %d detection images total", len(records))


if __name__ == "__main__":
    run_detection_pipeline(
        BASE_DIR / "images",
        BASE_DIR / "images" / "birds",
        log_file=BASE_DIR / "images" / "birds" / "processed_images.json",
    )
