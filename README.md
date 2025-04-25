# AutoSAM 2: A more efficient encoder

The single most important thing about Greenstand's use case is that the Encoder is highly parallelizable. This autosam2 offers a speedup of about 4.5 vis-à-vis autosam. Refer to [AutoSAM](https://github.com/xhu248/AutoSAM) here. 

You can read how SAM1's Image Encoder is not efficient for running inference at scale -> [Benchmarking README](sam1/README.md).

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. The model design is a simple transformer architecture with streaming memory for real-time video processing. 

## SAM 2 Installation

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```
If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[notebooks]"
```

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.5.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.5.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).

Please see [`INSTALL.md`](./INSTALL.md) for FAQs on potential issues and solutions.

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)

Then SAM 2 can be used in a few lines as follows for image and video prediction.

### Image prediction

SAM 2 has all the capabilities of [SAM](https://github.com/facebookresearch/segment-anything) on static images, and we provide image prediction APIs that closely resemble SAM for image use cases. The `SAM2ImagePredictor` class has an easy interface for image prompting.

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

Please refer to the examples in [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)) for static image use cases.

SAM 2 also supports automatic mask generation on images just like SAM. Please see [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)) for automatic mask generation in images.


For image prediction:

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

For video prediction:

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

## Model Description

### SAM 2.1 checkpoints

The table below shows the improved SAM 2.1 checkpoints released on September 29, 2024. We are currently using the base plus model.
|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2.1_hiera_tiny <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt))    |     38.9     |          91.2          |        76.5         |        71.8        |       77.3        |
|   sam2.1_hiera_small <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt))   |      46      |          84.8          |        76.6         |        73.5        |       78.3        |
| sam2.1_hiera_base_plus <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)) |     80.8     |        64.1          |        78.2         |        73.7        |       78.2        |
|   sam2.1_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt))   |    224.4     |          39.5          |        79.5         |        74.6        |       80.6        |

### SAM 2 checkpoints

The previous SAM 2 checkpoints released on July 29, 2024 can be found as follows:

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny <br /> ([config](sam2/configs/sam2/sam2_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt))   |     38.9     |          91.5          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small <br /> ([config](sam2/configs/sam2/sam2_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt))   |      46      |          85.6          |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus <br /> ([config](sam2/configs/sam2/sam2_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)) |     80.8     |     64.8    |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large <br /> ([config](sam2/configs/sam2/sam2_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt))   |    224.4     | 39.7 |        76.0         |        74.6        |       79.8        |

Speed measured on an A100 with `torch 2.5.1, cuda 12.4`. See `benchmark.py` for an example on benchmarking (compiling all the model components). Compiling only the image encoder can be more flexible and also provide (a smaller) speed-up (set `compile_image_encoder: True` in the config).
## Segment Anything Video Dataset

See [sav_dataset/README.md](sav_dataset/README.md) for details.

## Training SAM 2

You can train or fine-tune SAM 2 on custom datasets of images, videos, or both. Please check the training [README](training/README.md) on how to get started.

## Web demo for SAM 2

We have released the frontend + backend code for the SAM 2 web demo (a locally deployable version similar to https://sam2.metademolab.com/demo). Please see the web demo [README](demo/README.md) for details.

## License  
- **SAM Components**: [Apache 2.0](LICENSE-APACHE)  
- **AutoSAM 2 Modifications**: [Proprietary](LICENSE) (restrictions apply).  

⚠️ **You may NOT**:  
- Publish research using AutoSAM 2 without permission.  
- Use AutoSAM 2 commercially without a license.  

**Contact**: [neelanchal.gahalot@greenstand.org](mailto:neelanchal.gahalot@greenstand.org)  


## Citing AUTOSAM2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@software{Greenstand_AutoSAM2,
  author = {Gahalot, Neelanchal and Greenstand},
  title = {{AutoSAM 2: Efficient Segment Anything Modifications}},
  year = {2024},
  publisher = {Greenstand},
  url = {https://github.com/greenstand/autosam2},
  note = {Proprietary modifications; contact for publication use.}
}
```
