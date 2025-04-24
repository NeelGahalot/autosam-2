import os
import time
import random
import warnings
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dice_loss import SoftDiceLoss


from plant_dataset import LeafDataset
# Training transforms (based on your existing transforms)
import torchvision.transforms as transforms
from PIL import Image

from autosam_utils import *

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

train_img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_mask_transform = transforms.Compose([
    
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

def main():
    # Set up random seed for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 4
    num_workers = 2
    learning_rate = 0.0005
    epochs = 50
    print_freq = 10
    num_classes = 2  # Binary segmentation
    HOME = '/app'
    base_dir= f'{HOME}/inhouse'
    image_dir = os.path.join(base_dir, 'samples')
    mask_dir = os.path.join(base_dir, 'binary_masks')
    splits_dir = os.path.join(base_dir, 'splits')
    
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Splits directory: {splits_dir}")
    
    # Create dataset and dataloader
    dataset = LeafDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=os.path.join(splits_dir, 'train.txt'),
        img_transform=train_img_transform,
        mask_transform=train_mask_transform
    )

    # Create validation dataset
    val_dataset = LeafDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=os.path.join(splits_dir, 'val.txt'),
        img_transform=train_img_transform,
        mask_transform=train_mask_transform
    )

    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    build_sam_seg = build_sam_vit_h_seg_cnn
    # Model loading 
    #HOME = '/teamspace/studios/this_studio'
    sam2_checkpoint = f"{HOME}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    #build the model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_encoder = sam2_model.image_encoder.to(device)
    model = _build_sam_seg_model(sam2_encoder, 2).to(device)
    model = model.to(device)

    # Freeze image encoder weights
    for name, param in model.named_parameters():
        if param.requires_grad and "image_encoder" in name or "iou" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Loss functions
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    ce_loss = torch.nn.CrossEntropyLoss()

    # Create save directory
    save_dir = "output_experiment/leaf_segmentation"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train(train_loader, model, optimizer, dice_loss, ce_loss, epoch, device, print_freq, writer)
        # Validate
        val_loss = validate(val_loader, model, dice_loss, ce_loss, epoch, device, writer)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save model if validation loss improves
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(save_dir, 'checkpoints', 'model_best.pth'))
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(save_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))

    writer.close()
    print('Training completed')

def train(train_loader, model, optimizer, dice_loss, ce_loss, epoch, device, print_freq, writer):
    model.train()
    losses = []
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    end = time.time()

    for i, (img, label) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to device
        img = img.float().to(device)
        label = label.long().to(device)
        b, c, h, w = img.shape

        # Forward pass
        mask, iou_pred = model(img)
        #print(mask.shape)
        mask = mask.view(b, -1, h, w)
        iou_pred = iou_pred.squeeze().view(b, -1)

        # Calculate loss
        pred_softmax = F.softmax(mask, dim=1)
        loss = ce_loss(mask, label.squeeze(1)) + dice_loss(pred_softmax, label.squeeze(1))
        losses.append(loss.item())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to TensorBoard
        writer.add_scalar('train_loss', loss.item(), global_step=i + epoch * len(train_loader))

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {loss.item():.4f}')

    avg_loss = np.mean(losses)
    print(f'Epoch {epoch} - Average train loss: {avg_loss:.4f}')
    writer.add_scalar('epoch_train_loss', avg_loss, epoch)
    
    return avg_loss

def validate(val_loader, model, dice_loss, ce_loss, epoch, device, writer):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for i, (img, label) in enumerate(val_loader):
            # Move data to device
            img = img.float().to(device)
            label = label.long().to(device)
            b, c, h, w = img.shape

            # Forward pass
            mask, iou_pred = model(img)
            mask = mask.view(b, -1, h, w)

            # Calculate loss
            pred_softmax = F.softmax(mask, dim=1)
            loss = ce_loss(mask, label.squeeze(1)) + dice_loss(pred_softmax, label.squeeze(1))
            losses.append(loss.item())
            
    avg_loss = np.mean(losses)
    print(f'Validation - Epoch: {epoch}, Loss: {avg_loss:.4f}')
    writer.add_scalar('val_loss', avg_loss, epoch)
    
    return avg_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()