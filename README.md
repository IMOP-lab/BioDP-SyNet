# BioDP-SyNet: Biophysically Orchestrated Dual-Path Synergistic Network for Breast Tumour Segmentation

<!-- [](https://www.google.com/search?q=https://arxiv.org/abs/xxxx.xxxxx) [](https://opensource.org/licenses/MIT) -->

**BioDP-SyNet** is a novel, physics-informed deep learning framework designed for precise and robust breast tumour segmentation in Dynamic Contrast-Enhanced MRI (DCE-MRI). By integrating MRI governing equations and biophysical principles, BioDP-SyNet effectively addresses challenges such as intratumoral heterogeneity, imaging noise, and artefactual interference.

## Abstract

Highly sensitive to tissue haemodynamics, dynamic contrast-enhanced MRI (DCE-MRI) increasingly functions as the modality of choice for early breast-cancer screening and diagnosis, yet intratumoral heterogeneity, imaging noise and artefactual interference—coupled with CNNs’ constrained receptive fields that impede global-structure capture—erect formidable barriers to automated, precise tumour segmentation. Addressing these challenges, this work presents the **Biophysically Orchestrated Dual-Path Synergistic Network (BioDP-SyNet)**, wherein MRI-physical governing equations and physics-informed neural paradigms converge within parallel spatial-diffusion and frequency-fluctuation pathways.

  - The **spatial stream** embeds an **Edge-Preserving Explicit Diffusion (EPED) layer** for adaptive, iterative smoothing, abating signal variance and noise.
  - The **frequency stream** integrates **FreqDualis-HoloSchrod (FDH) Attention** and **Laplacian-Gradient (La-Gra) Attention** to emulate frequency interference and phase modulation, heightening sensitivity to textural nuance and local-frequency anomalies.

Evaluation on BreastDM and BCMedSet demonstrates BioDP-SyNet's gains in Dice, IoU and 95HD, with pronounced small-lesion detection and complex-boundary delineation.

-----

## Core Architecture

BioDP-SyNet employs a dual-encoder, single-decoder architecture with parallel paths that synergistically extract and enhance features from two complementary physical perspectives.BioDP-SyNet employs a dual-encoder, single-decoder architecture with parallel paths that synergistically extract and enhance features from two complementary physical perspectives.

![BioDP-SyNet Architecture](https://github.com/IMOP-lab/BioDP-SyNet/raw/main/Picture/BioDP-SyNet.png)

*Figure: Architectural schematic of BioDP-SyNet, detailing its dual-encoder framework where the Spatial Diffusion Path (top) and Frequency Fluctuation Path (bottom) synergize to fuse multi-scale features into a shared decoder.*

### 1\. Spatial Diffusion Path

This path focuses on suppressing image noise and smoothing the highly heterogeneous signals within tumour regions.

![Edge-Preserving Explicit Diffusion Layer](https://github.com/IMOP-lab/BioDP-SyNet/raw/main/Picture/EPED.png)

  - **EPED (Edge-Preserving Explicit Diffusion) Layer**: The core module, which abstracts feature maps as manifolds and evolves them using an adaptive Partial Differential Equation (PDE). It effectively removes high-frequency noise while preserving critical tumour boundary structures via a content-aware diffusion coefficient.

### 2\. Frequency Fluctuation Path

This path is designed to overcome the local receptive field limitations of traditional convolutions, capturing macroscopic tumour morphology and fine boundary geometry.

![FreqDualis-HoloSchrod Attention](https://github.com/IMOP-lab/BioDP-SyNet/raw/main/Picture/FDH.png)

  - **FDH (FreqDualis-HoloSchrod) Attention**: Inspired by the Schrödinger equation from quantum mechanics, this module operates in the frequency domain to capture global shape information and phase consistency, crucial for holistic morphological assessment.

![Laplacian-Gradient Attention](https://github.com/IMOP-lab/BioDP-SyNet/raw/main/Picture/La-Gra.png)

  - **La-Gra (Laplacian-Gradient) Attention**: Embeds fixed differential operators (Laplacian and gradient) as explicit physical priors into the attention mechanism. This enables the model to better distinguish true anatomical boundaries from artifacts and enhances its sensitivity to subtle boundary details.



-----

## Research Highlights

  - 🧠 **Dual-Physics Guided**: Proposes a novel dual-physics guided network (BioDP-SyNet) for breast MRI segmentation.
  - 🌀 **Spatial Diffusion Denoising**: The spatial stream embeds a diffusion model to abate signal noise and variance.
  - 🌊 **Frequency Path Enhancement**: Frequency path attentions enhance global morphology and boundary sensitivity.
  - 🎯 **Superior Performance**: Demonstrates superior segmentation of small lesions and complex boundaries.

-----

## Experimental Results

BioDP-SyNet was benchmarked against 13 baseline models on two challenging DCE-MRI datasets.

### Qualitative Comparison

BioDP-SyNet accurately delineates tumour contours, demonstrating superior robustness compared to other models, especially in difficult cases involving small lesions (Case 5) and complex backgrounds (Case 10).

![Qualitative Comparison](https://github.com/IMOP-lab/BioDP-SyNet/raw/main/Picture/comparison_of_models.png)

*Figure: A qualitative comparison against 13 leading models on representative cases. BioDP-SyNet (last column) shows the highest consistency with the expert-annotated Ground Truth.*

### Quantitative Results

BioDP-SyNet achieves state-of-the-art performance across all key evaluation metrics.

**On BreastDM Dataset**:

| Model           | Dice(%) $\uparrow$     | IOU(%) $\uparrow$      | 95HD $\downarrow$      | Kappa(%) $\uparrow$    | MCC (%) $\uparrow$     |
| ------------------- | ----------------- | ----------------- | --------------- | ----------------- | ----------------- |
| U-Net               | 70.50 ± 32.26     | 62.20 ± 31.61     | 1.94 ± 0.89     | 70.45 ± 32.28     | 71.52 ± 31.75     |
| SegNet              | 76.62 ± 26.04     | 67.53 ± 26.23     | 1.87 ± 0.84     | 76.56 ± 26.09     | 77.40 ± 25.65     |
| ENet                | 68.62 ± 30.07     | 58.70 ± 28.67     | 2.00 ± 0.75     | 68.57 ± 30.08     | 69.78 ± 29.49     |
| R2U-Net             | 76.26 ± 28.53     | 68.06 ± 28.34     | 1.83 ± 0.87     | 76.21 ± 28.56     | 77.11 ± 28.06     |
| UNeXt               | 61.04 ± 29.38     | 49.59 ± 29.41     | 2.21 ± 0.80     | 60.99 ± 29.42     | 63.15 ± 28.57     |
| MEWUNet             | *77.77 ± 24.73*   | *68.58 ± 24.91*   | *1.82 ± 0.68*   | *77.72 ± 24.77*   | *78.42 ± 24.33*   |
| PAttUNet            | 71.84 ± 33.88     | 64.69 ± 33.07     | 1.88 ± 0.87     | 71.80 ± 33.90     | 72.74 ± 33.32     |
| DAttUNet            | 72.64 ± 31.87     | 64.72 ± 31.15     | 1.91 ± 0.83     | 72.60 ± 31.90     | 73.38 ± 31.59     |
| PolypPVT            | 73.69 ± 22.49     | 62.22 ± 22.22     | 2.00 ± 0.65     | 73.63 ± 22.52     | 74.51 ± 22.01     |
| MDViT               | 62.53 ± 28.75     | 51.07 ± 27.13     | 2.12 ± 0.71     | 62.46 ± 28.77     | 63.98 ± 28.18     |
| VM-UNet             | 54.75 ± 33.95     | 44.75 ± 30.55     | 2.28 ± 0.81     | 54.69 ± 33.95     | 56.57 ± 33.53     |
| VMamba              | 49.36 ± 34.92     | 39.80 ± 30.71     | 2.39 ± 0.85     | 49.30 ± 34.92     | 51.46 ± 34.65     |
| CCViM               | 65.73 ± 31.60     | 56.00 ± 30.39     | 2.05 ± 0.83     | 65.67 ± 31.64     | 67.09 ± 31.06     |
| **BioDP-Sy (Ours)** | **79.79 ± 24.26** | **71.37 ± 25.25** | **1.69 ± 0.73** | **79.75 ± 24.30** | **80.56 ± 23.66** |

**On BCMedSet Dataset**:

| Model           | Dice(%) $\uparrow$     | IOU(%) $\uparrow$      | 95HD $\downarrow$      | Kappa(%) $\uparrow$    | MCC (%) $\uparrow$     |
| ------------------- | ----------------- | ----------------- | --------------- | ----------------- | ----------------- |
| U-Net               | 80.04 ± 25.89     | 72.41 ± 27.03     | 2.10 ± 1.16     | 79.91 ± 25.92     | 78.86 ± 27.70     |
| SegNet              | 79.87 ± 25.58     | 72.01 ± 26.61     | 2.11 ± 1.48     | 79.70 ± 25.65     | *79.18 ± 26.68*   |
| ENet                | 75.31 ± 24.58     | 65.07 ± 24.20     | 2.31 ± 1.12     | 75.14 ± 24.60     | 74.62 ± 25.74     |
| R2U-Net             | 78.65 ± 27.29     | 70.90 ± 27.78     | 2.13 ± 1.19     | 78.51 ± 27.31     | 77.98 ± 28.39     |
| UNeXt               | 78.36 ± 23.60     | 69.02 ± 24.23     | 2.23 ± 1.05     | 78.22 ± 23.62     | 77.25 ± 25.35     |
| MEWUNet             | 79.46 ± 27.58     | 72.23 ± 28.19     | *2.06 ± 1.18*   | 79.33 ± 27.63     | 78.66 ± 28.59     |
| PAttUNet            | 79.92 ± 26.77     | 72.62 ± 27.82     | 2.07 ± 1.15     | 79.79 ± 26.81     | 78.75 ± 28.43     |
| DAttUNet            | *80.45 ± 25.61*   | *72.84 ± 26.60*   | 2.08 ± 1.15     | *80.31 ± 25.65*   | 78.71 ± 28.06     |
| PolypPVT            | 69.74 ± 21.89     | 57.03 ± 21.25     | 2.48 ± 0.98     | 69.54 ± 21.87     | 69.42 ± 23.83     |
| MDViT               | 75.41 ± 26.37     | 65.90 ± 25.94     | 2.25 ± 1.81     | 75.25± 26.39      | 74.23 ± 27.84     |
| VM-UNet             | 79.34 ± 27.44     | 72.07 ± 28.44     | 2.06 ± 1.24     | 79.20 ± 27.50     | 78.13 ± 28.97     |
| VMamba              | 79.91 ± 24.05     | 71.30 ± 24.35     | 2.19 ± 1.10     | 79.76 ± 24.09     | 78.00 ± 26.89     |
| CCViM               | 78.89 ± 27.40     | 71.40 ± 28.22     | 2.08 ± 1.15     | 78.75 ± 27.46     | 78.15 ± 28.28     |
| **BioDP-Sy (Ours)** | **81.30 ± 26.86** | **74.68 ± 28.00** | **2.04 ± 1.21** | **81.17 ± 26.89** | **79.26 ± 29.41** |

-----

## Installation & Setup

**1. Clone this repository:**

```bash
git clone https://github.com/IMOP-lab/BioDP-SyNet.git
cd BioDP-SyNet
```

**2. Create and activate a Conda environment:**

```bash
conda create -n biodp python=3.9
conda activate biodp
```

Key dependencies include `torch>=2.0.0` and `torchvision`.

-----

## Usage

### Dataset Preparation

  - **BreastDM**: This is a publicly available dataset. Please download it from the [official source](https://doi.org/10.1016/j.compbiomed.2023.107255) and organize it as required.
  - **BCMedSet**: This is a private, multi-institutional dataset and is not publicly available.

Place your datasets in a `data/` directory or specify the path in the training/testing commands.

### Training

```bash
python train.py --dataset BreastDM --data_path /path/to/your/dataset --epochs 50 --batch_size 4 --lr 1e-4
```

### Testing

Download our pre-trained model weights (link to be provided) and place them in the `weights/` folder.

```bash
python test.py --dataset BreastDM --data_path /path/to/your/dataset --weights ./weights/biodp_synet_best.pth
```

-----

## License

This project is licensed under the [MIT License](https://github.com/IMOP-lab/BioDP-SyNet/blob/main/LICENSE).

## Acknowledgments

We thank the creators and contributors of the [BreastDM](https://doi.org/10.1016/j.compbiomed.2023.107255) dataset. We also extend our gratitude to Sir Run Run Shaw Hospital and The Second Affiliated Hospital, Zhejiang University School of Medicine for their support in collecting the BCMedSet dataset.