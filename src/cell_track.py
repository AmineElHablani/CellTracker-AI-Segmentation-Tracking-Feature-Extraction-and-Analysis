import numpy as np

# from cell_tracker import track_cells
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from scipy.spatial.distance import cdist

import numpy as np, cv2, os
from matplotlib import cm
import pandas as pd 
import shutil

from tqdm import tqdm
import math


def iou(bb1, bb2):
    """Compute IoU between two bounding boxes"""
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    return inter_area / float(bb1_area + bb2_area - inter_area + 1e-6)

def compute_bbox(mask):
    props = regionprops(mask)
    return [prop.bbox for prop in props], [prop.centroid for prop in props]

def mask_to_objects(mask):
    props = regionprops(mask)
    return [
        {'id': None, 'label': prop.label, 'centroid': prop.centroid, 'bbox': prop.bbox, 'area': prop.area}
        for prop in props
    ]

def track_cells(mask_list, iou_threshold=0.3):
    next_id = 1
    tracks = {}  # {id: [{frame, centroid, area}]}

    prev_objects = []

    for t, mask in enumerate(mask_list):
        curr_objects = mask_to_objects(mask)

        if t == 0:
            for obj in curr_objects:
                obj['id'] = next_id
                tracks[next_id] = [{'frame': t, **obj}]
                next_id += 1
        else:
            cost_matrix = np.ones((len(prev_objects), len(curr_objects))) * np.inf

            for i, prev in enumerate(prev_objects):
                for j, curr in enumerate(curr_objects):
                    cost_matrix[i, j] = 1 - iou(prev['bbox'], curr['bbox'])

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < (1 - iou_threshold):
                    curr_objects[j]['id'] = prev_objects[i]['id']
                    tracks[prev_objects[i]['id']].append({'frame': t, **curr_objects[j]})
                    assigned.add(j)

            for j, curr in enumerate(curr_objects):
                if j not in assigned:
                    curr['id'] = next_id
                    tracks[next_id] = [{'frame': t, **curr}]
                    next_id += 1

        prev_objects = curr_objects

    return tracks

def detect_mitosis(tracks, distance_threshold=20):
    mitosis_events = []

    track_endings = {}
    track_beginnings = {}

    for tid, dets in tracks.items():
        track_endings[tid] = dets[-1]
        track_beginnings[tid] = dets[0]

    for mother_id, end_det in track_endings.items():
        frame_end = end_det['frame']
        end_centroid = np.array(end_det['centroid'])

        # Find candidates starting in the next frame
        daughters = []
        for cand_id, start_det in track_beginnings.items():
            if cand_id == mother_id:
                continue
            if start_det['frame'] == frame_end + 1:
                dist = np.linalg.norm(end_centroid - np.array(start_det['centroid']))
                if dist < distance_threshold:
                    daughters.append((cand_id, dist))

        if len(daughters) >= 2:
            mitosis_events.append({
                'mother_id': mother_id,
                'frame': frame_end,
                'daughter_ids': [d[0] for d in daughters[:2]]
            })

    return mitosis_events

def build_instance_map(prob_masks, thr=0.35, min_px=5):
    """
    Convert sigmoid probability masks to a single labelled image.

    Parameters
   
    prob_masks : torch.FloatTensor  (N, H, W)
        Output of `pred["masks"].squeeze(1)`.
    thr : float
        Pixel threshold. 0.35–0.5 works well for Mask‑R‑CNN nuclei.
    min_px : int
        Minimum area (in pixels) to keep a mask.  Set to 0 or ≤5 to
        keep all nuclei; raise if you want to drop tiny speckles.

    Returns
    -------
    inst_map : np.ndarray  (H, W)  dtype uint16
        0 = background, 1..K = nucleus IDs.
    """
    H, W = prob_masks.shape[-2:]
    inst_map = np.zeros((H, W), np.uint16)
    inst_id = 1

    for pm in prob_masks:                       # pm is (H, W) float32
        mask = (pm > thr).cpu().numpy()
        if min_px and mask.sum() < min_px:
            continue
        inst_map[mask] = inst_id
        inst_id += 1

    return inst_map


def save_labeled_masks(tracks,
                       original_masks,
                       save_dir="labeled_masks_with_ids"):
    """
    Parameters
   
    tracks : dict
        {track_id: [ {'frame': int, 'label': int, 'centroid': (r,c)}, ... ]}
    original_masks : list[np.ndarray]
        List of instance masks (H, W) for every frame.
    save_dir : str
        Folder where colour PNGs are written.
    """
    os.makedirs(save_dir, exist_ok=True)
    cmap = cm.get_cmap("tab20", 256)            # fixed colour palette

    # lookup tables for quick access
    label2tid   = {(d['frame'], d['label']): tid
                   for tid, dets in tracks.items() for d in dets}
    label2cent  = {(d['frame'], d['label']): d['centroid']
                   for tid, dets in tracks.items() for d in dets}

    for f_idx, mask in enumerate(original_masks):
        h, w = mask.shape
        coloured = np.zeros((h, w, 3), np.uint8)

        # iterate over every nucleus in the frame
        for lbl in np.unique(mask):
            if lbl == 0:
                continue

            # deterministic colour: based on label so it doesn't flicker
            colour = (np.array(cmap(lbl % 256)[:3]) * 255).astype(np.uint8)
            coloured[mask == lbl] = colour

            key = (f_idx, lbl)
            if key in label2tid:               # draw ID only if tracked
                tid = label2tid[key]
                cy, cx = label2cent[key]
                cx, cy = int(cx), int(cy)
                cv2.putText(coloured, str(tid), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(save_dir, f"labeled_{f_idx:03d}.png"),
                    coloured[:, :, ::-1])      # RGB → BGR

    print(f"✅ Labeled masks saved to “{save_dir}”.")



def extract_advanced_features(tracks, mitosis_events=[], pixel_size=1.0, frame_interval=1.0):
    mitosis_mothers = {m['mother_id'] for m in mitosis_events}
    mitosis_daughters = {d for m in mitosis_events for d in m['daughter_ids']}

    feature_table = []

    for track_id, detections in tracks.items():
        sizes = [d['area'] for d in detections]
        frames = [d['frame'] for d in detections]
        centroids = np.array([d['centroid'] for d in detections])

        speeds = []
        angles = []
        accelerations = []

        for i in range(1, len(centroids)):
            p1 = np.array(centroids[i - 1])
            p2 = np.array(centroids[i])
            dx, dy = p2[1] - p1[1], p2[0] - p1[0]

            dist = np.linalg.norm(p2 - p1)
            speed = (dist * pixel_size) / frame_interval
            speeds.append(speed)

            angle = math.degrees(math.atan2(-dy, dx)) % 360  # up is 90
            angles.append(angle)

        for i in range(1, len(speeds)):
            accel = (speeds[i] - speeds[i - 1]) / frame_interval
            accelerations.append(accel)

        feature_table.append({
            'id': track_id,
            'birth_frame': frames[0],
            'death_frame': frames[-1],
            'lifetime': len(frames),
            'mean_size': np.mean(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'mean_speed': np.mean(speeds) if speeds else 0,
            'min_speed': np.min(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'mean_acceleration': np.mean(accelerations) if accelerations else 0,
            'min_acceleration': np.min(accelerations) if accelerations else 0,
            'max_acceleration': np.max(accelerations) if accelerations else 0,
            'mean_angle': np.mean(angles) if angles else 0,
            'min_angle': np.min(angles) if angles else 0,
            'max_angle': np.max(angles) if angles else 0,
            'total_distance': np.sum(speeds) * frame_interval if speeds else 0,
            'mitosis_status': track_id in mitosis_mothers or track_id in mitosis_daughters,
            'role': ('mother' if track_id in mitosis_mothers else
                     'daughter' if track_id in mitosis_daughters else 'none')
        })

    return feature_table

def save_features_to_csv(features, filename="cell_features.csv"):
    df = pd.DataFrame(features)
    df.to_csv(filename, index=False)
    print(f"[✔] Cell features saved to: {filename}")
    return df 
    
    
    

def extract_images_from_video(
        video_path: str,
        output_dir: str = "frames",
        basename: str = "frame",
        ext: str = ".tif",
        zero_pad: int = 4,
        start_index: int = 0,
        every_nth: int = 1
    ):
    """
    Extract frames from *video_path* and save them to *output_dir* as
    basename####.ext (zero‑padded) in the order they appear.

    Parameters
   
    video_path : str
        Path to the source video (e.g. "tracked_cells.mp4").
    output_dir : str, default "frames"
        Directory where frames will be written. Created if it doesn't exist.
    basename : str, default "frame"
        Prefix for saved images.
    ext : str, default ".png"
        Image extension (".png", ".jpg", ".tif", ...).
    zero_pad : int, default 4
        Number of digits to pad the frame counter with.
    start_index : int, default 0
        First index in the filename sequence.
    every_nth : int, default 1
        Save one frame out of *every_nth* (e.g. 5 → keep 0,5,10,…).
    """

    # setup
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = start_index

    # extraction loop
    with tqdm(total=frame_total, desc="Extracting") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:                      # end of video
                break

            # save only every_nth frame
            if idx % every_nth == 0:
                name = f"{basename}{idx:0{zero_pad}d}{ext}"
                cv2.imwrite(os.path.join(output_dir, name), frame)

            idx += 1
            pbar.update(1)

    cap.release()
    print(f"[✔] Saved {idx - start_index} frames to '{output_dir}/'")
    
    







def create_video_from_images(
        image_dir,
        output_file="tracked_cells.mp4",
        fps=5,
        return_bytes=False
):
    import cv2, os, tempfile
    from tqdm import tqdm

    image_files = sorted(os.listdir(image_dir))
    if not image_files:
        raise ValueError("No images found in " + image_dir)

    first = cv2.imread(os.path.join(image_dir, image_files[0]))
    h, w = first.shape[:2]

    # trying an H.264 fourcc first, fall back to mp4v if unavailable
    for fourcc_str in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        video  = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
        if video.isOpened():
            print(f"Using codec {fourcc_str}")
            break
    else:
        raise RuntimeError("Could not open VideoWriter with any supported codec")

    for f in tqdm(image_files, desc="Encoding video"):
        frame = cv2.imread(os.path.join(image_dir, f))
        video.write(frame)
    video.release()
    print("✔ video saved to", output_file)

    if return_bytes:
        with open(output_file, "rb") as f:
            return f.read()        # return raw bytes for st.video
    return output_file            # path on disk








def save_extracted_images(temp_image_dir, dest_dir):
    """
    Copy every image from *temp_image_dir* into <dest_root>/<uuid>/ and return that
    new folder path.

    Parameters
   
    temp_image_dir : str | Path
        Temporary folder holding freshly‑extracted images.
    dest_root : str
        Base directory under which a unique subfolder will be created, e.g.
        "dashboard_results/raw_frames".
    """
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    os.makedirs(dest_dir, exist_ok=True)

    for root, _, files in os.walk(temp_image_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in valid_ext:
                src = os.path.join(root, fname)
                shutil.copy2(src, os.path.join(dest_dir, fname))

    print(f"📂  {len(os.listdir(dest_dir))} images persisted to: {dest_dir}")
    return dest_dir  # returns plain string path


def create_video_from_raw_images(
    image_dir,
    output_file="raw_video.mp4",
    fps=5,
    return_bytes=False,
    enhance_contrast=False,    
):
    """
    Build a video from colour (or gray) images in *image_dir*.
    • Keeps original colours (no grayscale conversion).
    • Optionally performs CLAHE contrast‑boost per frame.
    """
    #list all files that look like images
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg",
                               ".tif", ".tiff", ".bmp"))
    ])
    if not image_files:
        raise ValueError(f"No valid images found in {image_dir}")

    #probe first frame for dimensions
    first = cv2.imread(os.path.join(image_dir, image_files[0]))   # colour read
    if first is None:
        raise RuntimeError("Could not read first image")
    h,  w  = first.shape[:2]

    # open VideoWriter (best codecs first)
    for fourcc_str in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        video  = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
        if video.isOpened():
            print(f"[✔] Using codec: {fourcc_str}")
            break
    else:
        raise RuntimeError("Cannot open VideoWriter with any supported codec")

    #encode
    for fname in tqdm(image_files, desc="Encoding raw video"):
        img_path = os.path.join(image_dir, fname)
        frame    = cv2.imread(img_path)
        if frame is None:
            continue

        # enhancement (CLAHE on the L channel)
        if enhance_contrast:
            lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b  = cv2.split(lab)
            clahe    = cv2.createCLAHE(clipLimit=2.0,
                                       tileGridSize=(8, 8))
            l2       = clahe.apply(l)
            frame    = cv2.cvtColor(cv2.merge((l2, a, b)),
                                    cv2.COLOR_LAB2BGR)

        #guarantee same size (in case some frames differ)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h),
                               interpolation=cv2.INTER_AREA)

        video.write(frame)

    video.release()
    print("✅ Raw video saved to", output_file)

    if return_bytes:
        with open(output_file, "rb") as f:
            return f.read()
    return output_file

