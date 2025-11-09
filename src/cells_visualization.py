from skimage import io                     # keeps original .tif intensity
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from PIL import Image
import numpy as np
from cell_track import *
from glob import glob 
from stqdm import stqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# safe TIFF reader
def safe_read(path):
    with Image.open(path) as im:
        return np.array(im)


def preprocess_frame(img_np):
    # 2–98 % contrast stretch
    p2, p98 = np.percentile(img_np, (2, 98))
    img_np  = np.clip((img_np - p2) / (p98 - p2 + 1e-3), 0, 1)
    # grayscale → 3‑ch
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, -1)          # H,W,3
    # to tensor C,H,W
    return torch.from_numpy(img_np).permute(2,0,1).float()


def build_instance_map(prob_masks, thr=0.35, min_px=40):
    inst = np.zeros(prob_masks.shape[-2:], np.uint16)
    inst_id = 1
    for pm in prob_masks:                 # pm : (H,W) float32
        m = (pm > thr).cpu().numpy()
        if m.sum() >= min_px:
            inst[m] = inst_id
            inst_id += 1
    return inst

def safe_imread(path):
    """
    Read TIFF regardless of the compression plugin availability.
    Falls back to Pillow if skimage + imagecodecs is missing.
    """
    try:
        return io.imread(path)              #fast path
    except ValueError as e:
        if "imagecodecs" in str(e):
            return np.array(Image.open(path)) 
        raise                


def preprocess_fluo(img_np):
    """
    - Percentile stretch (2..98%)
    - Convert grayscale → 3‑channel
    - Return torch tensor in (C,H,W), float32, 0‑1
    """
    p2, p98 = np.percentile(img_np, (2, 98))
    img_np = np.clip((img_np - p2) / (p98 - p2 + 1e-3), 0, 1)

    if img_np.ndim == 2:                      # grayscale
        img_np = np.stack([img_np] * 3, -1)   # H,W,C → 3‑ch

    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()  # C,H,W
    return img_t
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def visualize_cleared_and_prediction(model,image_folder,device=device):
    image_files  = sorted([f for f in os.listdir(image_folder) if f.endswith(".tif")])

    for idx, fname in enumerate(tqdm(image_files, desc="Visualising")):
        # Read TIFF with fallback
        img_np = safe_imread(os.path.join(image_folder, fname))

        # identical preprocessing to training/val
        img = preprocess_fluo(img_np).to(device)           # (3,H,W), 0‑1
        with torch.no_grad():
            pred = model([img])[0]

        masks = (pred["masks"] > 0.5).squeeze(1).cpu()     # (N,H,W) bool
        img3  = img.cpu()                                  # already 3‑ch

        # Draw overlay
        overlay = draw_segmentation_masks(
            (img3 * 255).byte(), masks, alpha=0.6)
        

        # side‑by‑side figure
        fig, axes = plt.subplots(1, 3, figsize=(13, 13))
        
        # Tensor -> CPU -> NumPy (take first channel for grayscale)
        img_gray = img.cpu()[0].numpy()             # (H,W)  float32  0‑1
        axes[0].imshow(img_gray, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        clear_img = (1.0 - img_gray) * 255.0        # still NumPy, fine for imshow
        axes[1].imshow(clear_img, cmap="gray")
        axes[1].set_title("Cleared")
        axes[1].axis("off")
        
        # overlay is already on CPU
        axes[2].imshow(overlay.permute(1, 2, 0))
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        plt.tight_layout()





def predict_masks(model,out_dir,image_folder):


    img_files = sorted(
        [p for p in image_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {'.tif', '.png', '.jpg', '.jpeg'}]
    )
    for i, path in enumerate(stqdm(img_files, desc="Segmenting")):
        if i <=10:
            print(path)
        # read + preprocess
        frame_np = safe_read(path)
        frame_t  = preprocess_frame(frame_np).to(device)  # (3,H,W)

        # inference
        with torch.no_grad():
            pred = model([frame_t])[0]

        if pred["masks"].numel() == 0:
            inst_map = np.zeros(frame_np.shape[:2], np.uint16)
        else:
            prob = pred["masks"].squeeze(1)              # (N,H,W) float
            inst_map = build_instance_map(prob)

        # save
        np.save(os.path.join(out_dir, f"mask_{i:03d}.npy"), inst_map)
        vis = (inst_map.astype(np.uint8) * 25).clip(0, 255)
        cv2.imwrite(os.path.join(out_dir, f"mask_vis_{i:03d}.png"), vis)

        print(f"✓ Frame {i+1}/{len(img_files)} done")

    print("✅ All instance masks exported to", out_dir)



def safe_imread(path):
    try:
        return io.imread(path)
    except ValueError as e:
        if "imagecodecs" in str(e):
            return np.array(Image.open(path))
        raise

def preprocess_fluo(img_np):
    p2, p98 = np.percentile(img_np, (2, 98))
    img_np  = np.clip((img_np - p2) / (p98 - p2 + 1e-3), 0, 1)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, -1)
    return img_np  # H,W,3 float32 (0-1)

def colorize_instance_mask(mask):
    """Return an RGB image where each non‑zero label gets a random color."""
    if mask.ndim != 2:
        raise ValueError("mask should be 2‑D")
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    ids = np.unique(mask)
    ids = ids[ids != 0]  # skip background
    rng = np.random.RandomState(42)
    for inst_id in ids:
        color = rng.rand(3)
        rgb[mask == inst_id] = color
    return rgb




def visuzalize_mask_and_overlay(img_dir,mask_folder,n_viz=35):
    image_files  = sorted(glob(os.path.join(img_dir, "*.tif")))
    mask_files   = sorted(glob(os.path.join(mask_folder, "mask_*.npy")))

    # display first 4 examples
    n_show = min(n_viz, len(image_files))

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3 * n_show))

    for idx in range(n_show):
        img_path  = image_files[idx]
        mask_path = mask_files[idx]

        img_np   = preprocess_fluo(safe_imread(img_path))
        mask_np  = np.load(mask_path)

        overlay  = (0.7 * img_np + 0.3 * colorize_instance_mask(mask_np))

        axes[idx, 0].imshow(img_np, cmap='gray')
        axes[idx, 0].set_title(f"Original {idx}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(colorize_instance_mask(mask_np))
        axes[idx, 1].set_title("Mask")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay")
        axes[idx, 2].axis('off')

    plt.tight_layout()





def safe_imread(path):
    """
    Read TIFF regardless of the compression plugin availability.
    Falls back to Pillow if skimage + imagecodecs is missing.
    """
    try:
        return io.imread(path)              #fast path
    except ValueError as e:
        if "imagecodecs" in str(e):
            return np.array(Image.open(path))
        raise



def preprocess_fluo(img_np):
    """
    - Percentile stretch (2..98%)
    - Convert grayscale → 3‑channel
    - Return torch tensor in (C,H,W), float32, 0‑1
    """
    p2, p98 = np.percentile(img_np, (2, 98))
    img_np = np.clip((img_np - p2) / (p98 - p2 + 1e-3), 0, 1)

    if img_np.ndim == 2:                      # grayscale
        img_np = np.stack([img_np] * 3, -1)   # H,W,C -> 3‑ch

    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()  # C,H,W
    return img_t
