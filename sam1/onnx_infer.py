import torch
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import time
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold, normalize
import onnxruntime

from segment_anything.utils.onnx import SamOnnxModel

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    
class ONNXSamPredictor:
    def __init__(self, sam_predictor: SamPredictor, onnx_model_path: str):
        self.predictor = sam_predictor
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None):
        # Prepare input coordinates
        if point_coords is None and box is None:
            raise ValueError("Must specify either point_coords or box")
            
        if box is not None:
            box_coords = np.array([[box[0], box[1]], [box[2], box[3]]])
            box_labels = np.array([2, 3])
            
            if point_coords is not None:
                onnx_coord = np.concatenate([point_coords, box_coords], axis=0)
                onnx_label = np.concatenate([point_labels, box_labels], axis=0)
            else:
                onnx_coord = box_coords
                onnx_label = box_labels
        else:
            onnx_coord = point_coords
            onnx_label = point_labels
            
        # Add a batch index and padding point
        onnx_coord = np.concatenate([onnx_coord, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([onnx_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
        
        # Transform coordinates
        onnx_coord = self.predictor.transform.apply_coords(onnx_coord[0], self.predictor.original_size).astype(np.float32)[None, ...]
        
        # Prepare mask input
        if mask_input is None:
            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            onnx_mask_input = mask_input.astype(np.float32)
            onnx_has_mask_input = np.ones(1, dtype=np.float32)
            
        # Get image embedding
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        
        # Package inputs
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.predictor.original_size, dtype=np.float32)
        }
        
        # Run inference
        masks, _, _ = self.ort_session.run(None, ort_inputs)
        masks = masks > self.predictor.model.mask_threshold
        
        return masks[0]


model_type = 'vit_l'
checkpoint = '/teamspace/studios/this_studio/greenstand_segmentation_model_build/segmentation_model_build/AutoSAM/scripts/sam_vit_l_0b3195.pth'
device = 'cuda:0'
# greenstand_segmentation_model_build/segmentation_model_build/segment-anything/scripts/sam_onnx.onnx
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
onnx_model_path = "scripts/sam_onnx.onnx"
predictor = SamPredictor(sam_model)
onnx_predictor = ONNXSamPredictor(predictor, onnx_model_path)
print(onnx_predictor)
start = time.time()
image_path = "truck.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
    
    # Example point input
input_box = np.array([400, 600, 700, 875])
    
    # Predict mask

masks = onnx_predictor.predict(box=input_box)
end = time.time()
print(f'prediction took {end-start}')
plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
plt.savefig('onnx_pred.jpg')