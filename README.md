# ODP-Net: Operator-Driven Propagation Network for Breast Tumor Segmentation in DCE-MRI

This repository contains the network implementation supporting **ODP-Net**, a physics-inspired, operator-driven dual-path architecture for breast tumor segmentation in dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI).

ODP-Net incorporates mathematically motivated operators as architectural priors for feature learning. It does **not** impose residual losses from a known, closed-form governing equation for breast tumor morphology. The model is designed to jointly address local signal heterogeneity, limited global context, and boundary ambiguity.

## Overview

ODP-Net uses a dual-encoder, single-decoder design. Features from a spatial diffusion path and a frequency fluctuation path are fused at each scale and decoded into the final segmentation mask.

- **Spatial diffusion path:** applies the **Globally-Modulated Diffusion (GMD)** layer at each encoder stage. GMD uses an adaptive, Laplacian-based diffusion process to regularize local features while preserving useful boundary structure.
- **Frequency fluctuation path:** uses **Frequency-Domain Propagator (FDP) attention** in shallower stages to promote global feature interaction and morphological representation.
- **Boundary-aware feature learning:** applies **Differential Operator Priors (DOP) attention** at the deepest frequency-encoder stage. DOP combines fixed gradient and Laplacian operators with attention features to provide geometric priors for boundary representation.
- **Decoder:** concatenates the paired multi-scale encoder features and progressively upsamples them to produce the segmentation logits.

## Repository Layout

```text
Networks/
  main.py                  ODP-Net model definition
  encoder.py               Residual frequency-path feature extractor
  simple_conv_blocks.py    Convolutional encoder/decoder blocks
  residual.py              Residual building blocks
  helper.py                Encoder utilities
  regularization.py        Regularization utilities
PINNs/
  *.py                     Standalone PDE/PINN reference examples
Picture/                   Earlier architecture and module illustrations
```

### Source-code terminology

The manuscript uses the finalized R5 terminology below. Some source identifiers and figures retain their earlier names; the following mapping is provided for traceability.

| R5 manuscript term | Current implementation identifier |
| --- | --- |
| GMD layer | `EPEDLayer` / `diffusion_attn*` |
| FDP attention | `HoloschrodAtt` / `freq_attn*` |
| DOP attention | `LaplacianGradientAttention` / `combined_attention` |
| ODP-Net | `DP_CoNet` |

The model implementation is located in [`Networks/main.py`](Networks/main.py).

## Experimental Setting

ODP-Net was trained and evaluated on two DCE-MRI cohorts:

| Dataset | Training slices | Validation slices | Test slices | Availability |
| --- | ---: | ---: | ---: | --- |
| BreastDM | 20,432 | 1,989 | 7,089 | Public; see the [dataset publication](https://doi.org/10.1016/j.compbiomed.2023.107255) |
| BCMedSet | 5,856 | 672 | 831 | Private multi-institutional cohort |

All evaluated networks were trained for 50 epochs under a common configuration: Python 3.9, PyTorch 2.0.0, CUDA 11.8, and an initial learning rate of `1e-4`. The reported inference benchmark used a `256 x 256` three-channel input on an NVIDIA GeForce RTX 3080 GPU.

Because the architecture is two-dimensional, predicted slices were reconstructed into three-dimensional patient volumes before evaluation. Results are reported as patient-level median (Q1-Q3). Statistical comparisons used one-sided Wilcoxon signed-rank tests with Benjamini-Hochberg false-discovery-rate correction; `*` denotes `p < 0.05` for the comparison with ODP-Net.

## Results

### BreastDM

| Model | 3D Dice (%) | 3D IoU (%) | 3D 95HD (voxels) | 3D Kappa (%) | 3D MCC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 76.32 (63.32-86.71)* | 61.71 (46.33-76.53)* | 6.05 (2.24-85.09) | 76.29 (63.25-86.68)* | 76.57 (66.39-86.96)* |
| SegNet | 78.86 (68.83-85.31) | 65.10 (52.47-74.38) | 9.55 (3.50-78.98) | 78.82 (68.78-85.26) | 79.75 (70.13-85.30) |
| ENet | 70.50 (56.25-83.34)* | 54.44 (39.13-71.44)* | 5.16 (2.24-36.66) | 70.44 (56.18-83.32)* | 71.30 (59.40-83.60)* |
| R2U-Net | 79.37 (72.79-87.08)* | 65.80 (57.22-77.11) | 5.11 (2.24-80.82) | 79.33 (72.64-87.04) | 79.48 (73.78-87.39) |
| UNeXt | 68.38 (55.96-75.13)* | 51.95 (38.85-60.16)* | 7.00 (2.83-62.07) | 68.34 (55.93-75.10)* | 69.02 (58.09-76.26)* |
| MEWUNet | 78.63 (63.37-87.64) | 64.78 (46.38-78.00)* | 4.85 (1.80-81.38) | 78.58 (63.30-87.62)* | 78.82 (65.15-87.69)* |
| PAttUNet | 80.82 (54.09-87.82)* | 67.82 (37.08-78.28)* | **4.29 (2.00-82.51)** | 80.77 (53.91-87.78)* | 80.96 (56.13-87.85)* |
| DAttUNet | 78.18 (54.47-87.28)* | 64.20 (37.45-77.44)* | 4.85 (2.24-82.13) | 78.13 (54.42-87.27)* | 78.43 (58.29-87.53)* |
| PolypPVT | 74.96 (60.56-82.49)* | 59.95 (43.43-70.20)* | 8.54 (2.00-88.26) | 74.90 (60.49-82.47)* | 75.28 (60.95-82.54)* |
| MDViT | 66.56 (44.20-76.67)* | 49.88 (28.37-62.16)* | 71.38 (5.50-93.20) | 66.51 (44.18-76.58)* | 68.71 (47.22-77.71)* |
| VM-UNet | 64.01 (28.73-72.51)* | 47.10 (16.85-56.88)* | 44.60 (6.10-90.68) | 63.96 (28.68-72.48)* | 65.74 (36.42-72.88)* |
| VMamba | 54.51 (26.84-69.04)* | 37.51 (15.50-52.71)* | 39.25 (5.02-87.79) | 54.45 (26.79-68.99)* | 59.41 (34.84-69.69)* |
| CCViM | 69.84 (45.66-80.89) | 53.66 (29.65-67.91)* | 72.18 (4.33-96.42) | 69.77 (45.48-80.86) | 71.29 (48.99-80.95) |
| **ODP-Net** | **81.14 (70.00-88.18)** | **68.27 (53.85-78.86)** | 4.62 (2.24-86.54) | **81.08 (69.99-88.14)** | **81.65 (70.85-88.23)** |

### BCMedSet

| Model | 3D Dice (%) | 3D IoU (%) | 3D 95HD (voxels) | 3D Kappa (%) | 3D MCC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 80.30 (78.45-83.88) | 67.09 (64.55-72.23) | **2.24 (2.00-9.22)** | 80.08 (78.30-83.81) | 80.67 (78.40-83.88) |
| SegNet | 80.82 (77.16-86.60) | 67.81 (62.82-76.36) | 2.83 (2.00-8.60) | 80.65 (77.04-86.55) | 80.80 (77.38-86.55) |
| ENet | 76.28 (72.68-81.88)* | 61.66 (57.08-69.32)* | 5.83 (2.24-6.40) | 76.04 (72.11-81.81)* | 76.22 (73.58-81.86)* |
| R2U-Net | 80.59 (76.33-83.97) | 67.50 (61.72-72.37) | 3.00 (2.00-10.82) | 80.51 (76.13-83.77) | 80.62 (76.19-83.87) |
| UNeXt | 79.84 (75.40-85.73) | 66.44 (60.52-75.03) | 3.16 (2.00-5.00) | 79.64 (74.92-85.55) | 79.65 (76.41-85.70) |
| MEWUNet | 82.64 (76.17-85.34) | 70.41 (61.51-74.42) | **2.24 (2.00-4.47)** | 82.50 (76.09-85.16) | 82.50 (76.29-85.20) |
| PAttUNet | 81.40 (74.89-86.12) | 68.63 (59.86-75.62) | 3.61 (2.00-5.48) | 81.17 (74.35-86.07) | 81.44 (75.11-86.18) |
| DAttUNet | 81.44 (77.24-86.42) | 68.69 (62.92-76.09) | 3.61 (2.00-7.21) | 81.26 (77.12-86.38) | 81.28 (77.28-86.57) |
| PolypPVT | 80.27 (76.61-83.38) | 67.04 (62.09-71.50) | 3.06 (1.73-5.10) | 80.12 (76.49-83.20) | 80.21 (77.19-83.20) |
| MDViT | 80.53 (74.88-85.42) | 67.41 (59.85-74.55) | 2.91 (1.41-4.36) | 80.39 (74.76-85.26) | 80.44 (75.42-85.24) |
| VM-UNet | 82.82 (74.09-86.63) | 70.68 (58.84-76.41) | 2.83 (2.00-6.32) | 82.62 (74.05-86.58) | 82.63 (74.05-86.81) |
| VMamba | 80.14 (73.80-83.13) | 66.86 (58.47-71.13) | 3.61 (2.00-7.42) | 79.90 (73.72-83.08) | 79.94 (74.09-83.38) |
| CCViM | 82.02 (72.71-86.37) | 69.53 (57.13-76.02) | 3.61 (2.00-5.00) | 81.88 (72.50-86.33) | 81.89 (72.83-86.40) |
| **ODP-Net** | **83.75 (80.12-85.35)** | **72.04 (66.84-74.44)** | 2.83 (1.41-10.27) | **83.69 (79.93-85.24)** | **84.25 (79.94-85.24)** |

Across the two within-dataset evaluations, ODP-Net achieved the highest overlap and consistency metrics among the evaluated methods. The qualitative analysis in the manuscript indicates improved depiction of micro-lesions and complex boundaries, with fewer glandular false positives. These findings do not establish resilience to controlled noise or artifact perturbations, because such targeted tests were not conducted. Diagnostic accuracy, treatment-planning contour quality, treatment-margin selection, and clinical outcomes were also not directly evaluated.

## Ablation Studies

On BreastDM, removing the operator-driven modules reduced the median Dice from 81.14% for the complete model to 74.50% for the architecture without GMD, FDP, or DOP. Individual module ablations support complementary contributions from local feature regularization (GMD), global morphological representation (FDP), and boundary-aware geometric priors (DOP).

| Configuration | Dice (%) | IoU (%) | 95HD (voxels) | Kappa (%) | MCC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| No GMD, FDP, or DOP | 74.50 (55.09-86.17) | 59.36 (38.02-75.69) | 73.56 (8.55-95.60) | 74.43 (55.04-86.15) | 74.94 (57.38-86.38) |
| GMD + FDP + DOP (ODP-Net) | **81.14 (70.00-88.18)** | **68.27 (53.85-78.86)** | **4.62 (2.24-86.54)** | **81.08 (69.99-88.14)** | **81.65 (70.85-88.23)** |

## Reproducibility Notes

This repository currently provides the network components and standalone PDE/PINN reference scripts. Dataset preparation, training orchestration, evaluation scripts, checkpoints, and the private BCMedSet data are not included. The code should therefore be treated as an implementation reference rather than a complete, turnkey training pipeline.

The included `PINNs/` examples illustrate conventional PINN formulations. They are not called by `DP_CoNet` during segmentation training; ODP-Net incorporates operator-inspired computations directly in the network architecture.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

We thank the contributors to the [BreastDM](https://doi.org/10.1016/j.compbiomed.2023.107255) dataset. We also acknowledge Sir Run Run Shaw Hospital and The Second Affiliated Hospital, Zhejiang University School of Medicine for support in collecting BCMedSet.

## Contact

For questions, contact [gaopeng.huang@hdu.edu.cn](mailto:gaopeng.huang@hdu.edu.cn) or [guohui71@sxmu.edu.cn](mailto:guohui71@sxmu.edu.cn).
