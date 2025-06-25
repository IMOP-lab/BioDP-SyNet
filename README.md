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

![Qualitative Comparison](https://github.com/IMOP-lab/BioDP-SyNet/raw/main/Picture/BioDP-SyNet.png)

*Figure: Architectural schematic of BioDP-SyNet, detailing its dual-encoder framework where the Spatial Diffusion Path (top) and Frequency Fluctuation Path (bottom) synergize to fuse multi-scale features into a shared decoder.*

### 1\. Spatial Diffusion Path

This path focuses on suppressing image noise and smoothing the highly heterogeneous signals within tumour regions.

  - **EPED (Edge-Preserving Explicit Diffusion) Layer**: The core module, which abstracts feature maps as manifolds and evolves them using an adaptive Partial Differential Equation (PDE). It effectively removes high-frequency noise while preserving critical tumour boundary structures via a content-aware diffusion coefficient.

### 2\. Frequency Fluctuation Path

This path is designed to overcome the local receptive field limitations of traditional convolutions, capturing macroscopic tumour morphology and fine boundary geometry.

  - **FDH (FreqDualis-HoloSchrod) Attention**: Inspired by the Schrödinger equation from quantum mechanics, this module operates in the frequency domain to capture global shape information and phase consistency, crucial for holistic morphological assessment.
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

| Model | Dice(%) $\uparrow$ | IOU(%) $\uparrow$ | 95HD $\downarrow$ |
| :--- | :---: | :---: | :---: |
| U-Net | 70.50 | 62.20 | 1.94 |
| MEWUNet | 77.77 | 68.58 | 1.82 |
| **BioDP-SyNet (Ours)** | **79.79** | **71.37** | **1.69** |

**On BCMedSet Dataset**:

| Model | Dice(%) $\uparrow$ | IOU(%) $\uparrow$ | 95HD $\downarrow$ |
| :--- | :---: | :---: | :---: |
| DAttUNet | 80.45 | 72.84 | 2.08 |
| SegNet | 79.87 | 72.01 | 2.11 |
| **BioDP-SyNet (Ours)** | **81.30** | **74.68** | **2.04** |

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

  - **BreastDM**: This is a publicly available dataset. Please download it from the [official source](https://www.google.com/search?q=https://github.com/zhao-yongsheng/BreastDM) and organize it as required.
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

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

## Acknowledgments

We thank the creators and contributors of the [BreastDM](https://doi.org/10.1016/j.compbiomed.2023.107255) dataset. We also extend our gratitude to Sir Run Run Shaw Hospital and The Second Affiliated Hospital, Zhejiang University School of Medicine for their support in collecting the BCMedSet dataset.