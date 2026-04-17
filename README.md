# Partial-Face-Recognition
Going beyond the traditional face detection focus the facial recognition when it is partially visible.

<img width="430" height="381" alt="occlusion_demo" src="https://github.com/user-attachments/assets/300d59d6-42f8-411e-b252-1bc4f9457983" />


## What is the Problem Context?

The problem context centers on the critical failure points of traditional facial recognition technology exposed by the COVID-19 pandemic, specifically the inability to accurately identify individuals when faced with partial facial occlusions such as masks, helmets, or scarves


# Partial Face Recognition System

This repository contains a robust biometric identification system designed to recognize individuals even with significant facial occlusions (masks, helmets, scarves). Developed to address the limitations of traditional facial recognition exposed during the COVID-19 pandemic, this project leverages **MobileFaceNet** and **ArcFace Loss** to provide high-accuracy partial face matching.

## 🚀 Features
- **Automated Data Pipeline**: Includes scripts for dataset multiplication and stratified splitting.
- **Advanced Augmentation**: Specialized `RandomFaceOcclusion` and `RandomErasing` to simulate real-world obstructions.
- **High-Performance Architecture**: MobileFaceNet optimized for mobile/embedded devices without sacrificing accuracy.
- **Deep Metric Learning**: Implementation of ArcFace (Additive Angular Margin Loss) for superior feature discrimination.
- **Evaluation Suite**: Integrated tools for Verification (1:1), Identification (1:N), and t-SNE embedding visualization.

<img width="461" height="671" alt="Model Architecture drawio" src="https://github.com/user-attachments/assets/428a817e-e79d-4b3a-bb4b-f0e14cf8a266" />
---

## ML Pipeline

Machine learning pipeline is realy importatnt for this kind of project for track down smooth process

<img width="683" height="142" alt="Pipeline PFR" src="https://github.com/user-attachments/assets/bc160a8c-94e8-456b-b7f8-73e5e63a0d18" />


---

## 🏗️ System Architecture

### Model: MobileFaceNet
The system uses an inverted residual structure with depthwise separable convolutions to maintain a lightweight footprint.
- **Input Size**: 112x112
- **Embedding Size**: 512-dimensional feature vector.
- **Activation**: PReLU for better gradient flow.

### Loss: ArcFace
To ensure that embeddings of the same person are close together and different people are far apart, we use ArcFace Loss:
- **Scale (s)**: 64.0
- **Margin (m)**: 0.5

<img width="576" height="474" alt="Model Architecture" src="https://github.com/user-attachments/assets/76404658-e330-4b23-9e29-dd77a485f6c4" />
---

## 📂 Project Structure

### 1. Data Preprocessing
- `multiply_dataset`: Ensures a minimum number of images per identity (target multiplier: 7x) to improve training stability.
- `split_dataset`: Automatically creates Train, Val, and Test splits while handling identity consistency.

### 2. Training Pipeline
- **Optimizers**: Separate SGD optimizers for model weights and ArcFace class centers to ensure convergence.
- **Scheduler**: Cosine annealing with a 5-epoch warmup phase.
- **Augmentations**:
    - `RandomResizedCrop` (simulates partial views)
    - `ColorJitter` & `GaussianBlur` (illumination/blur invariance)
    - `RandomFaceOcclusion` (structured masks/blocks)

 

### 3. Evaluation & Testing
- **Verification**: Tests 1:1 matching with ROC-AUC analysis.
- **Identification**: Top-1 and Top-5 accuracy metrics for gallery searches.
- **Visualization**: t-SNE plots to analyze the clustering of facial embeddings.

<img width="933" height="389" alt="Screenshot 2026-04-15 215243" src="https://github.com/user-attachments/assets/75b2f14c-e455-4f7b-ab30-d2f4bbd48fbe" />
<img width="765" height="536" alt="Screenshot 2026-04-15 215404" src="https://github.com/user-attachments/assets/97015cd2-9fdf-4e8f-a012-ded3980e326c" />


---

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install torch torchvision tqdm pillow numpy matplotlib seaborn scikit-learn
```

### Training
1. Update the `source_dir` paths in the script to point to your dataset (e.g., CASIA-WebFace).
2. Run the training cell to generate `best_mobilefacenet.pth`.

### Inference
You can verify a partial face against an enrolled full face using the following logic:
```python
# Enrollment
full_embedding = get_embedding("path_to_full_face.jpg")

# Verification
test_embedding = get_embedding("path_to_masked_face.jpg")
similarity = cosine_similarity(full_embedding, test_embedding)

if similarity > 0.96:
    print("Match Confirmed")
```

---

## 📊 Results Summary
- **Target Metrics**: 
    - High TPR (True Positive Rate) at low FPR (False Positive Rate).
    - Robust clustering of identities in 2D space via t-SNE.
- **Validation**: Includes real-time visualization of similarity scores with color-coded match/no-match borders.

---

## 👥 Team (E25)
* **Perera** - Project Manager / Team Leader (Agile Planning & Preprocessing)
* **Kasun** - AI Architecture (Model Design & Tuning)
* **Omidu** - UI Designer (Streamlit Development & Ethical Framework)
* **Oishee** - Document Lead / Quality Assurer (JIRA & Sprint Planning)

---

## 📝 Project Management
The project followed the **Agile/Scrum** framework, managed via **JIRA** for sprint backlogs, retrospectives, and ROAM analysis for risk management.
