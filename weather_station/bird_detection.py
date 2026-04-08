import os
import glob
import json
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytz
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.ops import nms
import torchvision.transforms as transforms
import torchvision.models as models
import urllib.request

IMAGENET_LABELS_FILE = "imagenet_classes.txt"

# --- New: Image Preprocessing ---
def preprocess_image(img):
    """
    Enhance image contrast and equalize histogram.
    """
    img = ImageOps.autocontrast(img)
    img = ImageOps.equalize(img)
    return img

# --- Core Functions ---
def load_models():
    try:
        model_small = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m.pt', skip_validation=True, verbose=False)
        model_large = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5l.pt', skip_validation=True, verbose=False)
        return [model_small, model_large]
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        raise

def detect_objects(img, models, confidence_threshold, target_classes):
    """Run detection on the image and filter for target classes with sufficient confidence."""
    all_detections = []
    for model in models:
        try:
            results = model(img)
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            continue
        df = results.pandas().xyxy[0]
        df = df[(df['confidence'] >= confidence_threshold) & (df['name'].isin(target_classes))]
        all_detections.append(df)
    
    if not all_detections:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_detections, ignore_index=True)
    if combined_df.empty:
        return combined_df

    boxes = torch.tensor(combined_df[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
    scores = torch.tensor(combined_df['confidence'].values, dtype=torch.float32)
    keep_indices = nms(boxes, scores, 0.5)
    return combined_df.iloc[keep_indices.numpy()]

def annotate_image(img, detections, font):
    """
    Draw bounding boxes and labels on the image.
    """
    draw = ImageDraw.Draw(img)
    detection_info = []
    for _, row in detections.iterrows():
        box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        label = f"{row['name']}: {row['confidence']:.2f}"
        draw.rectangle(box, outline='red', width=2)
        draw.text((row['xmin'], row['ymax']), label, fill='red', font=font)
        detection_info.append({
            "class": row['name'],
            "confidence": row['confidence'],
            "xmin": row['xmin'],
            "ymin": row['ymin'],
            "xmax": row['xmax'],
            "ymax": row['ymax']
        })
    return img, detection_info

def post_to_instagram(detection_posts, output_dir, log_file, bot):
    try:
        with open(log_file, "r") as f:
            posted_images = json.load(f)
    except FileNotFoundError:
        posted_images = []
    
    for post in detection_posts:
        filename = os.path.basename(post["original_path"])
        if filename in posted_images:
            logging.info(f"{filename} already posted, skipping.")
            continue

        output_path = os.path.join(output_dir, filename)
        post["annotated_img"].save(output_path, format="JPEG")
        time_str = post["timestamp"].astimezone(pytz.timezone("America/Denver")).strftime("%Y-%m-%d %H:%M:%S %Z")
        detection_strs = []
        for d in post["detections"]:
            if d["class"].lower() == "bird":
                if "refined_bird_class" in d:
                    text = (f"Detected bird with {d['confidence']*100:.1f}% confidence, "
                            f"refined to {d['refined_bird_class']} with {d['refined_bird_confidence']*100:.1f}% confidence")
                else:
                    text = f"Detected bird with {d['confidence']*100:.1f}% confidence"
            else:
                if "refined_animal_class" in d:
                    text = (f"Detected {d['refined_animal_class']} with {d['refined_animal_confidence']*100:.1f}% confidence")
                else:
                    text = f"Detected {d['class']} with {d['confidence']*100:.1f}% confidence"
            detection_strs.append(text)
        caption = f"Detected: {', '.join(detection_strs)} at {time_str}"
        bot.upload_photo(output_path, caption=caption)
        logging.info(f"Posted {output_path} with caption: {caption}")
        posted_images.append(filename)
        with open(log_file, "w") as f:
            json.dump(posted_images, f)

def filter_detections(detections):
    # Additional filtering can be implemented here.
    return detections

def load_imagenet_labels():
    if os.path.exists(IMAGENET_LABELS_FILE):
        with open(IMAGENET_LABELS_FILE) as f:
            return [l.strip() for l in f]
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    labels = [l.decode().strip() for l in urllib.request.urlopen(url)]
    with open(IMAGENET_LABELS_FILE, "w") as f:
        f.write("\n".join(labels))
    return labels

def load_classifier():
    classifier = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    classifier.eval()
    return classifier

classifier_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def refine_bird_detection(img, bbox, classifier_model, imagenet_classes):
    cropped = img.crop(bbox)
    input_tensor = classifier_transform(cropped).unsqueeze(0)
    with torch.no_grad():
        output = classifier_model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    top_prob, top_idx = probabilities.topk(1)
    label = imagenet_classes[top_idx.item()]
    BIRD_KEYWORDS = ['bird', 'sparrow', 'robin', 'eagle', 'hawk', 'finch', 'owl', 'parrot', 'duck']
    if not any(keyword in label.lower() for keyword in BIRD_KEYWORDS):
        label = "bird"
        confidence = 0.5 * top_prob.item()
    else:
        confidence = top_prob.item()
    return label, confidence

def refine_animal_detection(img, bbox, classifier_model, imagenet_classes, target_classes):
    cropped = img.crop(bbox)
    input_tensor = classifier_transform(cropped).unsqueeze(0)
    with torch.no_grad():
        output = classifier_model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    top_probs, top_idxs = probabilities.topk(5)
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        label = imagenet_classes[idx.item()]
        if label.lower() in [t.lower() for t in target_classes]:
            return label, prob.item()
    label = imagenet_classes[top_idxs[0][0].item()]
    return label, top_probs[0][0].item()

def refine_detections(detection_posts, classifier_model, imagenet_classes, target_classes):
    for post in detection_posts:
        for detection in post["detections"]:
            bbox = (detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"])
            if detection["class"].lower() == "bird":
                refined_label, refined_conf = refine_bird_detection(post["original_img"], bbox, classifier_model, imagenet_classes)
                if refined_label.lower() != "bird":
                    detection["refined_bird_class"] = refined_label
                    detection["refined_bird_confidence"] = refined_conf
            else:
                refined_label, refined_conf = refine_animal_detection(post["original_img"], bbox, classifier_model, imagenet_classes, target_classes)
                detection["refined_animal_class"] = refined_label
                detection["refined_animal_confidence"] = refined_conf
    return detection_posts

# --- Helper Functions for Post Scheduling ---
LAST_POST_FILE = "last_post_time.txt"
POST_INTERVAL_HOURS = 4

def can_post(last_post_file=LAST_POST_FILE, interval_hours=POST_INTERVAL_HOURS):
    try:
        with open(last_post_file, "r") as f:
            last_post_str = f.read().strip()
            last_post_time = datetime.fromisoformat(last_post_str)
    except Exception:
        return True
    return datetime.now() - last_post_time >= timedelta(hours=interval_hours)

def update_last_post_time(last_post_file=LAST_POST_FILE):
    with open(last_post_file, "w") as f:
        f.write(datetime.now().isoformat())

def select_best_detection_post(detection_posts):
    best_post = None
    best_bird_conf = -1
    for post in detection_posts:
        bird_confidences = [d['confidence'] for d in post['detections'] if d['class'].lower() == "bird"]
        if bird_confidences:
            max_conf = max(bird_confidences)
            if max_conf > best_bird_conf:
                best_bird_conf = max_conf
                best_post = post
    if best_post is None:
        best_post = max(detection_posts, key=lambda post: len(post['detections']))
    return best_post

# --- Main Pipeline ---
def run_detection_pipeline(
    image_dir,
    output_dir,
    confidence_threshold=0.3,
    log_file="processed_images.json",
    bot=None,  # Instagram bot instance (if available)
    hours_back=3,
    target_classes=['bird', 'squirrel', 'cat', 'rabbit', 'fox']  # only animals
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv5 models.
    models_list = load_models()
    
    # Get all JPEG images.
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    timestamps = []
    for img in image_files:
        try:
            base = os.path.basename(img).replace(".jpg", "")
            ts = datetime.strptime(base, "%Y%m%d_%H%M%S")
            ts = pytz.UTC.localize(ts)
            timestamps.append((img, ts))
        except ValueError:
            continue
    if not timestamps:
        logging.info("No properly named images found.")
        return

    timestamps.sort(key=lambda x: x[1])
    now = timestamps[-1][1]
    plot_start = now - timedelta(hours=hours_back)
    new_images = [(img, ts) for img, ts in timestamps if ts >= plot_start]
    
    logging.info(f"Found {len(new_images)} images from the last {hours_back} hours.")
    
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 50)
        except Exception:
            logging.warning("Falling back to default font.")
            font = ImageFont.load_default()
    
    detection_posts = []
    
    for img_path, img_timestamp in tqdm(new_images, desc="Processing images"):
        try:
            img_orig = Image.open(img_path).convert("RGB").rotate(180, expand=True)
            img_proc = preprocess_image(img_orig.copy())
        except Exception as e:
            logging.warning(f"Skipping {img_path}: {e}")
            continue

        detections = detect_objects(img_proc, models_list, confidence_threshold, target_classes)
        detections = filter_detections(detections)
        if detections.empty:
            continue
        
        annotated_img, detection_info = annotate_image(img_orig.copy(), detections, font)
        detection_posts.append({
            "original_img": img_orig,
            "annotated_img": annotated_img,
            "timestamp": img_timestamp,
            "detections": detection_info,
            "original_path": img_path
        })
    
    # Print detection summary
    print(f"Processed {len(new_images)} images.")
    print(f"Found detections in {len(detection_posts)} images.")
    
    if not detection_posts:
        logging.info("No animal detections found in processed images.")
        return

    classifier_model = load_classifier()
    imagenet_classes = load_imagenet_labels()
    detection_posts = refine_detections(detection_posts, classifier_model, imagenet_classes, target_classes)
    
    best_post = select_best_detection_post(detection_posts)
    if best_post is None:
        logging.info("No suitable post found.")
        return

    # If an Instagram bot is provided, use it; otherwise, move (save) the annotated images.
    if bot is not None:
        # Uncomment the following lines to post using your Instagram bot.
        # post_to_instagram([best_post], output_dir, log_file, bot)
        # update_last_post_time()
        pass
    else:
        for post in detection_posts:
            filename = os.path.basename(post["original_path"])
            output_path = os.path.join(output_dir, filename)
            post["annotated_img"].save(output_path, format="JPEG")
            print(f"Moved {post['original_path']} to {output_path}")
    
    # Rename files flagged for removal.
    for filename in os.listdir(output_dir):
        if "REMOVE_ME" in filename:
            old_path = os.path.join(output_dir, filename)
            new_filename = filename.replace(".REMOVE_ME", "")
            new_path = os.path.join(output_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

    print("Pipeline complete. All tasks done.")

# Example usage:
# run_detection_pipeline("path/to/images", "path/to/output")
