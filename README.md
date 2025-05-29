# ⚡ Powerline Health Detection via Drone-Based AI  
### 🛠 AI System for Florida Power & Light (FPL)

*An applied project using modern machine learning to inspect powerline infrastructure from UAV video. Designed for real-time analysis and future edge deployment.*

---

## 🎯 Project Objective

Design an AI system that identifies visible defects in powerline infrastructure using drone video. While the current demo runs on a high-performance PC, it could simulate **edge deployment constraints** and includes a parallel lightweight model for future embedded applications. The system is engineered to tolerate real-world visual noise (glare, occlusion, fog) and support iterative model updates and fault logging.

---

## 📷 Datasets Used

| Dataset | Description |
|--------|-------------|
| **InsPLAD** | 10K+ UAV images with annotations for defects (e.g., corrosion, broken caps) across multiple components. |
| **STN PLAD** | 2,400 annotated images from multiple tower angles and weather conditions. |
| **TTPLA** | 1,100 high-res images of powerline structures for segmentation and detection. |
| **Recognizance-2** | Binary dataset: powerline vs. non-powerline frames. |
| **VSB Signal-Based** | Time-series discharge signals to support multimodal learning. |
| **Custom RT-DETR Dataset** | Drone footage with insulator annotations for normal vs. defective classification. |

---

## 🧠 Model Pipeline

### 1. 🧼 Data Processing
- Standardize to COCO format for RT-DETR training.
- Apply domain-specific augmentations (fog, glare, blur).
- Use diffusion models + CLIP for generating edge case samples.
- Self-supervised feature pretraining with **DINOv2** or **Masked Autoencoders (MAE)**.

### 2. 🏗️ Model Selection & Training
- **Primary Model**: **RT-DETR** for detection and instance-level localization.
- **Segmentation Models**: U-Net or Mask2Former.
- **Multi-task Training**: Combine detection, segmentation, and defect classification.
- Tune via **Optuna/HyperOpt**. Add adversarial robustness (FGSM, PGD).

### 3. 📐 Structural Postprocessing (Optional)
- Transform outputs into graph representations for anomaly analysis.
- Use **GNNs** to detect structural defects (e.g., missing insulators).
- Validate structure with **persistent homology**.

### 4. 🔌 Multimodal Fusion (Optional)
- Sync imagery with signal-based readings (e.g., discharge data).
- Prototype fusion models using **Perceiver IO** or cross-modal attention.

### 5. 🧪 Evaluation & Robustness
- Metrics: mAP, IoU, AUROC, DSC.
- Robustness testing: glare, fog, occlusion simulation.
- Use **Monte Carlo Dropout**, **Deep Ensembles** for uncertainty.
- Test OOD response using entropy and Outlier Exposure techniques.

---

## ⚙️ Off-Grid Simulation & Design Strategy

### 1. Deployment Simulation
- Full pipeline demonstrated on **desktop hardware (RTX 3080 Ti)**.
- Simulated edge deployment using constrained inference settings.
- Parallel lightweight model included to reflect Jetson-class performance.

### 2. Model Optimization
- Train heavy and light models concurrently (RT-DETR vs. MobileViT/YOLOv8n).
- Quantize lightweight models with QAT or post-training methods.
- Optimize with **TensorRT** or **TVM**; simulate RTSP ingestion via GStreamer.

### 3. Logging and Storage
- Store detection logs locally (JSON/CSV) with **timestamp and GPS**.
- On-device summaries emulate edge device behavior for post-flight reviews.

### 4. OTA Simulation
- Demonstrate patch updates via mock OTA scripts (simulated LoRaWAN).
- Integrate continual learning via **EWC** with misclassified samples.

---

## 🧠 Graduate-Level Learning Objectives

| Topic | Techniques |
|-------|------------|
| Detection/Segmentation | RT-DETR, Mask2Former, U-Net |
| Representation Learning | DINOv2, MAE |
| Robust Training | FGSM, PGD, adversarial augmentation |
| Structural Validation | GNN postprocessing, persistent homology |
| Data Efficiency | Diffusion augmentation, active learning |
| Uncertainty Modeling | MC Dropout, OOD detection |
| Deployment Simulation | Quantization, TensorRT, OTA scripting |

---

## 📦 Deliverables for FPL

- ✅ Full PC-based inference pipeline using **RT-DETR** and segmentation head.
- ✅ Lightweight compressed model for edge simulation and constraint tests.
- ✅ Structural anomaly module (GNN-based, optional).
- ✅ Web dashboard or terminal log viewer for defect logging and review.
- ✅ Simulated federated update and continual learning flow.
- ✅ Detailed technical report with benchmarks, tradeoffs, and failure analysis.

---

## 🧩 Known Limitations & Design Decisions

| Concern | Explanation |
|--------|-------------|
| **Model size vs. latency** | Full model runs on PC; lightweight model simulates edge conditions. |
| **No edge hardware used in demo** | Edge performance is emulated via constraints and separate models. |
| **Annotation cost** | Managed with CLIP, SAM, and active sampling. |
| **Multimodal fusion is experimental** | Sensor + vision integration planned but not production-ready. |
| **Power hardware omitted** | No solar or buzzer components; emphasis is on software logging. |
| **GNN module is optional** | Structural graph used as enhancement for advanced validation. |

---

## 🖥️ Demo System Requirements

- **CPU**: Intel i9-9900K  
- **RAM**: 16GB DDR4  
- **GPU**: NVIDIA RTX 3080 Ti  
- **Frameworks**: PyTorch, MMDetection, MMSeg, TensorRT  
- **Core Model**: **RT-DETR** + segmentation/refinement module

---

## 📬 Contact

Questions, feedback, or contributions? Open a GitHub [issue](https://github.com/your-repo/issues) or submit a pull request.

---
