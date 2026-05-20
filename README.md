# NeuroCanvas — Neural Style Transfer

> Transfer the artistic style of any painting to any photograph in real-time using **Adaptive Instance Normalization (AdaIN)**.
>
> Live demo : https://huggingface.co/spaces/SOMIL007/NeuroCanvas

---

## What is This?

NeuroCanvas is a deep learning project that performs **neural artistic style transfer**. Given a content image (photograph) and a style image (painting), the model generates a new image that preserves the content structure while applying the artistic texture, colors, and brushstroke patterns of the painting.

Unlike slow optimization-based methods, AdaIN achieves **real-time inference** by learning to transfer style in a single forward pass through the network.

**Based on:** Huang, X., and Belongie, S. — *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* — ICCV 2017 — https://arxiv.org/abs/1703.06868

---

## How It Works

```
Content Image --> VGG Encoder --> Content Features -->
                                                       AdaIN --> Decoder --> Stylized Image
Style Image   --> VGG Encoder --> Style Features  -->
```

**AdaIN Formula:**
```
AdaIN(c, s) = std(s) * ( (c - mean(c)) / std(c) ) + mean(s)
```

The style statistics (mean and std) from the painting are transferred onto the normalized content features. The decoder reconstructs this back into an image.

**Style Strength (Alpha):**
```
t = alpha x AdaIN(c, s) + (1 - alpha) x c
```
Alpha of 1.0 = full style. Alpha of 0.0 = original content unchanged.

---

## Project Structure

```
NeuroCanvas/
|
|-- app.py                        Flask web server
|-- train.py                      Training script
|-- test.py                       Command-line inference
|-- requirements.txt              Python dependencies
|
|-- utils/
|   |-- models.py                 VGGEncoder + Decoder
|   |-- utils.py                  AdaIN, dataset, transforms
|
|-- templates/
|   |-- index.html                Web UI structure
|
|-- static/
|   |-- css/style.css             All styles (dark responsive UI)
|   |-- js/app.js                 Upload, progress, result logic
|   |-- uploads/                  Auto-created by Flask
|   |-- results/                  Auto-created by Flask
|
|-- content_data/                 Demo content images (11 photos)
|-- style_data/                   Demo style images (18 paintings)
|
|-- experiments/
|   |-- experiment_gpu/
|       |-- decoder_20.pth        Trained decoder checkpoint
|       |-- optimizer_20.pth      Optimizer state (for resume)
|       |-- output_20.png         Sample grid from training
|
|-- vgg_normalised.pth            Pre-trained frozen VGG encoder
```

---

## Installation

```bash
git clone https://github.com/yourusername/NeuroCanvas.git
cd NeuroCanvas
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
tqdm
Pillow
flask>=3.0.0
```

---

## Running the Web App

```bash
python app.py
```

Open browser at `http://localhost:5000`

**Steps:**
1. Click the **Content** zone and upload any photograph
2. Click the **Style** zone and upload any painting
3. Adjust **Alpha** slider (style strength: 0.0 to 1.0)
4. Select **Output Resolution** (256 / 512 / 768)
5. Click **Transfer Style**
6. Download the result

---

## Command-Line Inference

**Single image pair:**
```bash
python test.py \
  --content content_data/chicago.jpg \
  --style style_data/la_muse.jpg \
  --decoder experiments/experiment_gpu/decoder_20.pth \
  --vgg vgg_normalised.pth \
  --output result.png
```

**Control style strength with alpha:**
```bash
python test.py \
  --content content_data/chicago.jpg \
  --style style_data/la_muse.jpg \
  --decoder experiments/experiment_gpu/decoder_20.pth \
  --vgg vgg_normalised.pth \
  --alpha 0.7 \
  --output result_70.png
```

**Batch — all content x all style combinations:**
```bash
python test.py \
  --content_dir content_data \
  --style_dir style_data \
  --decoder experiments/experiment_gpu/decoder_20.pth \
  --vgg vgg_normalised.pth \
  --output_dir results/
```

---

## Training

### Dataset Setup

**Content — MS-COCO val2017 (~800 MB, 5,000 images)**

Download from: `http://images.cocodataset.org/zips/val2017.zip`

Extract to `data/coco/val2017/`

**Style — Best Artworks of All Time (~2 GB, 8,000 paintings)**

Download from: `https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time`

Extract to `data/wikiart/`

### Full Training Command

```bash
python train.py \
  --content_dir data/coco/val2017 \
  --style_dir data/wikiart \
  --vgg vgg_normalised.pth \
  --experiment experiment_gpu \
  --epochs 20 \
  --batch_size 8 \
  --repeat 1 \
  --augment \
  --style_weight 10.0 \
  --num_workers 2 \
  --save_interval 1 \
  --log_interval 100
```

### Resume Training

```bash
python train.py \
  --content_dir data/coco/val2017 \
  --style_dir data/wikiart \
  --vgg vgg_normalised.pth \
  --experiment experiment_gpu \
  --epochs 40 \
  --batch_size 8 \
  --resume \
  --decoder_path experiments/experiment_gpu/decoder_20.pth \
  --optimizer_path experiments/experiment_gpu/optimizer_20.pth
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| --content_dir | content_data | Path to content images |
| --style_dir | style_data | Path to style images |
| --vgg | vgg_normalised.pth | Path to VGG weights |
| --experiment | experiment1 | Experiment folder name |
| --epochs | 2 | Number of epochs |
| --batch_size | 4 | Batch size |
| --lr | 1e-4 | Learning rate |
| --lr_decay | 5e-5 | Learning rate decay |
| --content_weight | 1.0 | Content loss weight |
| --style_weight | 10.0 | Style loss weight |
| --repeat | 1 | Repeat small dataset N times |
| --augment | False | Enable data augmentation |
| --num_workers | 0 | DataLoader workers |
| --save_interval | 1 | Save checkpoint every N epochs |
| --resume | False | Resume from checkpoint |
| --decoder_path | None | Decoder checkpoint (for resume) |
| --optimizer_path | None | Optimizer checkpoint (for resume) |

### Training on Google Colab (Free GPU)

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os

# Extract project
with zipfile.ZipFile('/content/drive/MyDrive/NeuroCanvas/NeuroCanvas.zip', 'r') as z:
    z.extractall('/content/')

# Extract datasets
with zipfile.ZipFile('/content/drive/MyDrive/NeuroCanvas/val2017.zip', 'r') as z:
    z.extractall('/content/NeuroCanvas/data/coco/')

with zipfile.ZipFile('/content/drive/MyDrive/NeuroCanvas/wikiart.zip', 'r') as z:
    z.extractall('/content/NeuroCanvas/data/')

# Start training
!cd /content/NeuroCanvas && python train.py \
  --content_dir data/coco/val2017 \
  --style_dir data/wikiart \
  --vgg vgg_normalised.pth \
  --experiment experiment_gpu \
  --epochs 20 \
  --batch_size 8 \
  --num_workers 2 \
  --augment \
  --save_interval 1
```

### Expected Loss Progress

| Epoch | Loss (approx) |
|---|---|
| 1 | ~45 - 50 |
| 5 | ~20 - 25 |
| 10 | ~10 - 12 |
| 20 | ~6 - 8 |
| 40 | ~4 - 5 |
| 60 | ~3 - 4 |

---

## Model Architecture

### VGG Encoder (Frozen)

Pre-trained VGG-19 truncated at relu4-1. Used to extract multi-scale features.

```
Input (3 x H x W)
      |
  enc_1  -->  relu1-1  (64ch)
      |
  enc_2  -->  relu2-1  (128ch)
      |
  enc_3  -->  relu3-1  (256ch)
      |
  enc_4  -->  relu4-1  (512ch)  <-- AdaIN applied here
```

Encoder weights are frozen. Only the decoder is trained.

### Decoder (Trained)

Mirror of encoder with upsampling:

```
512ch --> Conv+ReLU --> Upsample
      --> Conv+ReLU --> Upsample
      --> Conv+ReLU --> Upsample
      --> Conv --> Sigmoid
      --> Output (3 x H x W)   values in [0, 1]
```

### Loss Functions

**Content Loss** — MSE at relu4-1:
```
L_content = MSE(encoder(output)[relu4-1], AdaIN(c, s))
```

**Style Loss** — MSE of mean and std at all 4 encoder levels:
```
L_style = sum over layers:
            MSE(mean(encoder(output)), mean(encoder(style)))
          + MSE(std(encoder(output)),  std(encoder(style)))
```

**Total Loss:**
```
L = w_content x L_content + w_style x L_style
```

---

## GPU vs CPU

| Device | Per Epoch (5k images) | 20 Epochs |
|---|---|---|
| CPU only | 5 - 8 hours | ~5 days |
| Colab T4 GPU | 20 - 30 mins | ~8 hours |
| Colab P100 GPU | 15 - 25 mins | ~6 hours |

Training on CPU is not practical for more than 3 epochs. Use Google Colab.

---

## Fixes Applied

| File | Fix |
|---|---|
| utils/utils.py | sorted(os.listdir) for reproducibility; RepeatedDataset; augmentation |
| utils/models.py | Sigmoid at Decoder output; map_location=cpu on VGG load |
| train.py | num_workers argument; save_interval bug fixed; always saves on final epoch; per-iteration log |
| test.py | New file — single image and batch inference with alpha control |
| app.py | New file — Flask web server |
| static/css/style.css | New — full dark responsive UI |
| static/js/app.js | New — drag-drop, progress steps, result display |

---

## .gitignore

```
data/
__pycache__/
*.pyc
.vscode/
static/uploads/
static/results/
experiments/*/optimizer_*.pth
```

Keep vgg_normalised.pth out of git (80 MB). Host on Google Drive and add a download link in setup instructions.

---

## References

- Huang and Belongie (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. ICCV 2017. https://arxiv.org/abs/1703.06868
- Gatys, Ecker, and Bethge (2016). A Neural Algorithm of Artistic Style. https://arxiv.org/abs/1508.06576
- Simonyan and Zisserman (2014). Very Deep Convolutional Networks (VGG). https://arxiv.org/abs/1409.1556

---

## License

MIT License — free to use, modify, and distribute.
