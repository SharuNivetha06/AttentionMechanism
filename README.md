# Effect of Attention Type and Attention Position in a Pretrained ResNet50

## What the notebook does

1. **Setup:** pins `protobuf`, imports libraries, fixes the random seed (`42`) and checks for GPUs.
2. **Data:** loads `tf_flowers` (5 classes: daisy, dandelion, roses, sunflowers, tulips), resizes to 224×224 and makes a fixed **70 / 15 / 15** train / validation / test split that is reused for every experiment.
3. **Augmentation:** a single shared pipeline (horizontal flip, rotation 0.05, zoom 0.10, translation 0.05, contrast 0.10) plus the ResNet50 `preprocess_input`.
4. **Model builder:** `build_stage_model()` splits the pretrained ResNet50 into stage segments and inserts an attention block after any chosen stage. The backbone is frozen, followed by GlobalAveragePooling → Dropout → Dense(5, softmax).
5. **Hyperparameter tuning:** grid search over learning rate `{1e-3, 3e-4, 1e-4}` and dropout `{0.3, 0.5}` on the baseline. The best pair is reused for all attention models.
6. **Experiments:** attention type, position, multiple positions, order, kernel size and additional mechanisms (see below).
7. **Evaluation:** accuracy, macro precision / recall / F1, total and trainable parameters, inference time per image, confusion matrices and plots.
8. **Qualitative analysis:** Grad-CAM, channel-attention weights, spatial-attention maps, self-attention maps and attention maps at different stages.
9. **Export:** saves results CSVs, the selected hyperparameters and trained models.

---

## Attention modules implemented

| Module | Description |
|---|---|
| Channel attention / SE | Global average pool → Dense (reduction r = 16, ReLU) → Dense (sigmoid) → channel-wise scaling |
| Spatial attention | Channel-wise mean and max → concatenate → Conv (k×k, sigmoid) → spatial scaling |
| Channel → Spatial / Spatial → Channel | The two modules applied sequentially, in either order |
| Self-attention | 1×1 Conv Q/K/V, scaled dot-product softmax over H×W positions, residual connection |
| ECA | Global average pool → 1-D convolution across channels → sigmoid |
| CBAM | Channel attention (avg + max pooled shared MLP), followed by spatial attention |
| Coordinate attention | Separate height and width pooling, shared 1×1 conv, per-axis sigmoid gates |
| Combined (C+S+SA) | Channel → Spatial → Self-attention |

### Insertion positions

| Position | ResNet50 layer | Feature size |
|---|---|---|
| P1 | `conv2_block3_out` | 56×56×256 |
| P2 | `conv3_block4_out` | 28×28×512 |
| P3 | `conv4_block6_out` | 14×14×1024 |
| P4 | `conv5_block3_out` | 7×7×2048 |

---

## Experiments

| # | Experiment | Models compared |
|---|---|---|
| 0 | Baseline | ResNet50 without attention |
| 1 | Attention type (all at P3) | Channel, Spatial, Channel→Spatial, Spatial→Channel, Self-Attention, SE, ECA, CBAM, Coordinate, Combined (C+S+SA) |
| 2 | Attention position | Channel→Spatial at P1, P2, P3, P4 |
| 3 | Multiple positions | P3+P4 and P2+P3+P4 |
| 4 | Attention order | Channel→Spatial vs Spatial→Channel at P3 |
| 5 | Spatial kernel size | 3×3, 5×5, 7×7 (Channel→Spatial at P3) |

---

## Training protocol

| Setting | Value |
|---|---|
| Image size / batch size | 224×224 / 32 |
| Optimizer / loss | Adam / categorical cross-entropy |
| Epochs (tuning / attention runs) | 5 / 8 |
| Callbacks | EarlyStopping (patience 3, restores best weights), ReduceLROnPlateau (factor 0.3, patience 2), ModelCheckpoint |
| Backbone | Frozen (transfer learning). `FINE_TUNE_EPOCHS = 0`, so the optional fine-tuning step is skipped |
| Seed | 42 |

---

## Outputs

| File | Content |
|---|---|
| `attention_results.csv` | Metrics for every model (accuracy, macro P/R/F1, params, inference time) |
| `kernel_size_results.csv` | Spatial-kernel ablation results |
| `gradcam_coverage.csv` | Grad-CAM activation coverage for correct vs misclassified images |
| `selected_hyperparameters.json` | Seed, best learning rate and dropout, batch size, epochs, split |
| `*.keras` | Saved model checkpoints |

### Figures produced

- Sample images from all five classes
- Training and validation accuracy / loss curves for every model
- Bar charts: accuracy and macro F1 per attention mechanism
- Scatter plots: accuracy vs parameters and macro F1 vs inference time
- Confusion matrices (baseline and CBAM)
- Grad-CAM grids for correct and misclassified images
- Channel-attention weights (top-20 and distribution)
- Spatial-attention maps and overlays, including maps at P2, P3 and P4
- Self-attention maps (mean, max and centre-query)

---

## Sample results (from the lab report)

Test-set results from one run. Your numbers may differ slightly.

| Model | Position | Accuracy | Macro F1 | Params |
|---|---|---|---|---|
| Baseline | – | 0.8984 | 0.8966 | 23,597,957 |
| Channel | P3 | 0.8929 | 0.8912 | 23,730,117 |
| Spatial | P3 | 0.9056 | 0.9043 | 23,598,055 |
| Channel+Spatial | P3 | 0.9093 | 0.9074 | 23,730,215 |
| Self-Attention | P3 | 0.9074 | 0.9066 | 24,909,957 |
| SE | P3 | 0.9111 | 0.9090 | 23,730,117 |
| Coordinate | P3 | 0.9111 | 0.9095 | 23,796,677 |

Differences between the top models are small (about 1 percentage point) and may be within run-to-run variation.

---

## Requirements

- Python 3.9+
- `tensorflow` (GPU recommended), `tensorflow-datasets`
- `protobuf==6.33.6` (installed in the first cell)
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

```bash
pip install tensorflow tensorflow-datasets "protobuf==6.33.6" numpy pandas matplotlib scikit-learn
```

## How to run

1. Open the notebook in Jupyter, Colab or Kaggle with a GPU runtime.
2. Run the cells **in order**. Later cells depend on the `models`, `histories` and `predictions` dictionaries built earlier.
3. The first run downloads `tf_flowers` through TensorFlow Datasets.
4. Training all models takes a while, so reduce `EPOCHS_ATTENTION` for a quick test.

---


## References

1. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
2. Hu et al., *Squeeze-and-Excitation Networks*, TPAMI 2020.
3. Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018.
4. Wang et al., *ECA-Net: Efficient Channel Attention*, CVPR 2020.
5. Hou et al., *Coordinate Attention for Efficient Mobile Network Design*, CVPR 2021.
6. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
7. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017.
