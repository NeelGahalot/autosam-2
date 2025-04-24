import torch
from collections import OrderedDict
from autosam_utils import *
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import os
import time
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def load_model(checkpoint_path, device='cuda'):
    # Build SAM2 model and encoder
    HOME = '/teamspace/studios/this_studio'
    sam2_checkpoint = f"{HOME}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_encoder = sam2_model.image_encoder.to(device)
    
    # Build segmentation model
    model = _build_sam_seg_model(sam2_encoder, 2).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    
    # Strip "module." prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

class BinarySegMetrics():
    """
    Binary Segmentation
    """
    def __init__(self):
        # two classes (foreground and background)
        self.n_classes = 2
        self.confusion_matrix = np.zeros((2, 2))
        # self.threshold = 0.5  # Threshold for converting probabilities to binary predictions

    def _fast_hist(self, label_true, label_pred):
        # label_pred = label_pred >= self.threshold  # Binarize predictions
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            2 * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_results(self):
        """Returns accuracy score evaluation result for binary segmentation."""
        hist = self.confusion_matrix
        tn, fp, fn, tp = hist.ravel()
        
        # Metrics for foreground
        foreground_total = tp + fn
        foreground_acc = tp / foreground_total if foreground_total > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        iou_foreground = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        iou_background = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
        mean_iou = (iou_foreground + iou_background) / 2

        #overall_acc = np.diag(hist).sum() / hist.sum()

        return {
            "Foreground Acc": foreground_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "IoU Foreground": iou_foreground,
            "IoU Background": iou_background,
            "Mean IoU": mean_iou
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            string += "%s: %f\n" % (k, v)
        return string
        
def _build_sam_seg_model(
    image_encoder,  # Now passed in!
    num_classes,
    checkpoint=None,
):
    prompt_embed_dim = 256

    sam_seg = AutoSamSeg(
        image_encoder=image_encoder,
        seg_decoder=MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
        ),
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = {
            k: v for k, v in state_dict.items()
            if k in sam_seg.state_dict().keys() and 'iou' not in k and "mask_tokens" not in k
        }
        sam_seg.load_state_dict(loaded_keys, strict=False)
        print("loaded keys:", loaded_keys.keys())

    return sam_seg


def build_sam_vit_h_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )

def build_sam_vit_l_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


def build_sam_vit_b_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )

class SegmentationDataset(Dataset):
    def __init__(self, sample_dir, mask_dir):
        self.sample_dir = sample_dir
        self.mask_dir = mask_dir
        self.file_list = os.listdir(sample_dir)
        
        # Store full paths for verification
        self.image_paths = []
        self.mask_paths = []
        for filename in self.file_list:
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(sample_dir, filename)
            mask_path = os.path.join(mask_dir, f"{base_name}_binarymask.jpg")  # or .jpg
            
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
        
        # Verify all files exist
        self._verify_files()
        
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _verify_files(self):
        """Verify that all files exist"""
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.img_transform(img)
        
        # Load and process target mask - ensure it's uint8
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        mask_array = (mask_array > 128).astype(np.uint8)  # Binarize to 0/1
        
        # Convert to torch tensor and ensure proper type
        mask_tensor = torch.from_numpy(mask_array).long()  # Convert to int64
        
        return img_tensor, mask_tensor, os.path.basename(img_path)

def metrics_from_dataset(model, dataset_path, device='cuda', batch_size=4):
    metrics = BinarySegMetrics()
    sample_dir = os.path.join(dataset_path, 'samples')
    mask_dir = os.path.join(dataset_path, 'binary_masks')
    
    # Create dataset and dataloader
    dataset = SegmentationDataset(sample_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        img_tensors, masks, filenames = batch
        img_tensors = img_tensors.to(device)
        
        # Inference
        with torch.no_grad():
            mask_preds, _ = model(img_tensors)  # Assuming your model returns (mask_pred, iou_pred)
            # Reshape and process output
            b, c, h, w = img_tensors.shape
            mask_preds = mask_preds.view(b, -1, h, w)
            probs = torch.softmax(mask_preds, dim=1)
            preds = (probs[:, 1] > 0.5).long().cpu().numpy()  # Get class 1 predictions
        
        # Update metrics for each item in batch
        for i in range(len(masks)):
            target = masks[i].numpy()
            pred = preds[i]
            metrics.update(target.astype(np.int64), pred.astype(np.int64))
        
    return metrics.get_results()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to your trained model checkpoint
    checkpoint_path = "/teamspace/studios/this_studio/sam2/output_experiment/leaf_segmentation/checkpoints/checkpoint_epoch_1.pth"
    
    # Load model
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully")
    
    dataset_path = "/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/Deeplab/india_sam_dino_annotations_large"
    if os.path.exists(dataset_path):
        print(f"\nEvaluating model on dataset: {dataset_path}")
        start=time.time()
        results = metrics_from_dataset(model, dataset_path, device, batch_size=4)
        end=time.time()
        print(end-start)
        print(results)
    else:
        print(f"\nDataset path not found: {dataset_path} - skipping evaluation")