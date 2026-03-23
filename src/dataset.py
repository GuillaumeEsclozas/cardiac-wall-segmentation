"""
CAMUS Dataset and DataModule for cardiac structure segmentation.

Handles data loading, patient-level fold splitting, and augmentation
for 2D echocardiographic images from the CAMUS dataset.
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
import nibabel as nib


# ══════════════════════════════════════════════════════════════
# Augmentation
# ══════════════════════════════════════════════════════════════

def get_train_transforms():
    """Affine augmentation only : flips, scale, translate, rotate."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
    ])


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class CAMUSDataset(Dataset):
    """
    CAMUS echocardiography dataset.

    Args:
        samples: List of (image_path, mask_path) tuples.
        img_size: Target image size (square).
        augment: Whether to apply training augmentations.
    """

    def __init__(self, samples, img_size=256, augment=False):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment
        self.transform = get_train_transforms() if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load NIfTI
        img = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)

        # Handle 3D (take first slice)
        if img.ndim == 3:
            img = img[:, :, 0]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # GT rounding fix (keeps labels clean after resize)
        mask = np.round(mask).astype(np.int64)
        mask = np.clip(mask, 0, 3)

        # Normalize image to [0, 1]
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Augmentation
        if self.augment and self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # To tensor
        img = torch.FloatTensor(img).unsqueeze(0)   # (1, H, W)
        mask = torch.LongTensor(mask.copy())          # (H, W)

        return img, mask


# ══════════════════════════════════════════════════════════════
# Data Discovery + Fold Splitting
# ══════════════════════════════════════════════════════════════

def discover_patient_samples(data_dir):
    """Scan dataset and return {patient_id: [(img_path, mask_path), ...]}."""
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, 'patient*')))
    print(f"Total patient folders found: {len(patient_dirs)}")

    patient_samples = {}
    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir)
        samples = []
        gt_files = sorted(glob.glob(os.path.join(pdir, '*_gt.nii.gz')))
        for gt_path in gt_files:
            img_path = gt_path.replace('_gt.nii.gz', '.nii.gz')
            if os.path.exists(img_path):
                samples.append((img_path, gt_path))
        if samples:
            patient_samples[patient_id] = samples

    total = sum(len(s) for s in patient_samples.values())
    print(f"Total samples: {total} across {len(patient_samples)} patients "
          f"({total / len(patient_samples):.1f} samples/patient)")
    return patient_samples


def get_fold_splits(patient_samples, n_folds=5, seed=42):
    """
    Patient-level k-fold split.

    Returns:
        List of (train_sample_list, val_sample_list) per fold.
    """
    patient_ids = sorted(patient_samples.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    fold_size = len(patient_ids) // n_folds
    folds = []

    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_folds - 1 else len(patient_ids)

        val_pids = patient_ids[val_start:val_end]
        train_pids = patient_ids[:val_start] + patient_ids[val_end:]

        train_samples = [s for pid in train_pids for s in patient_samples[pid]]
        val_samples = [s for pid in val_pids for s in patient_samples[pid]]

        folds.append((train_samples, val_samples))
        print(f"  Fold {fold_idx}: {len(train_pids)} train patients ({len(train_samples)} samples) | "
              f"{len(val_pids)} val patients ({len(val_samples)} samples)")

    return folds


# ══════════════════════════════════════════════════════════════
# DataModule
# ══════════════════════════════════════════════════════════════

class CAMUSDataModule(pl.LightningDataModule):
    """Simplified DataModule — takes pre-split sample lists, no internal splitting."""

    def __init__(self, train_samples, val_samples, batch_size=4,
                 img_size=256, augment=False, num_workers=4):
        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CAMUSDataset(
            self.train_samples, self.img_size, augment=self.augment
        )
        self.val_dataset = CAMUSDataset(
            self.val_samples, self.img_size, augment=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)
