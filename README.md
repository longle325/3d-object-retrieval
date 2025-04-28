# 3D Object Retrieval Pipeline for ROOMELSA Grand Challenge 2025

[![ROOMELSA Grand Challenge](https://img.shields.io/badge/SHREC-2025-blue)](https://aichallenge.hcmus.edu.vn/shrec-2025/smart3droom)

> This repository hosts the official implementation of our two-stage 3D object retrieval pipeline developed by team Stubborn_Strawberries. It demonstrates our approach for translating natural language queries and spatial context into accurate retrieval of 3D models, combining state-of-the-art vision–language embeddings, image captioning, and re-ranking techniques.

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🗂️ Repository Structure](#️-repository-structure)
- [🚀 Pipeline Overview](#-pipeline-overview)
  - [1. 3D Model Rendering](#1-3d-model-rendering)
  - [2. Vision–Language Embedding](#2-visionlanguage-embedding)
  - [3. Caption Generation](#3-caption-generation)
  - [4. Retrieval Process](#4-retrieval-process)
- [⚙️ Installation](#️-installation)
- [🎯 Usage](#-usage)
  - [Data Preparation](#data-preparation)
  - [Generating Embeddings](#generating-embeddings)
  - [Training BLIP-2](#training-blip-2)
  - [Inference](#inference)
- [📈 Results](#-results)
- [🎓 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- **Two-Stage Retrieval**: Combines fast candidate retrieval with refined caption-based re-ranking.
- **Multi-View Rendering**: Generates 20 evenly spaced views per 3D object for comprehensive embedding.
- **State-of-the-Art Models**:
  - **SIGLIP** for vision–language embeddings
  - **BLIP-2 (Flan-T5-xl)** for image captioning
  - **BGE-M3** for text embedding and final ranking
- **Scalable Storage**: Uses Milvus vector database with a FLAT index for efficient similarity search.

---

## 🗂️ Repository Structure

```
├── model/
│   ├── __init__.py          # Package initialization
│   ├── BGE_M3.py            # BGE-M3 text embeddings interface
│   ├── BLIP_model.py        # BLIP-2 image captioning implementation
│   └── SIGLIP_model.py      # SIGLIP vision–language embedding interface
├── convert_file_obj_to_glb.py       # OBJ → GLB converter
├── convert_glb_to_2d_imgs.py        # 3D → 2D view renderer
├── embed_2d_imgs.py                 # Embed rendered views into Milvus
├── infer_without_caption_n_rerank.py   # Stage 1 retrieval only
├── infer_with_caption_n_rerank.py      # Full two-stage retrieval
├── make_dataset_for_blip.py          # Prepare BLIP-2 training data
├── train_blip.py                     # BLIP-2 fine-tuning script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview

```

---

## 🚀 Pipeline Overview

A high-level view of the ROOMELSA two-stage retrieval pipeline:

![Pipeline image](https://drive.google.com/file/d/1MXGKE4STiCHEaIG_TNUNZsw2o3xvinv8)

### 1. 3D Model Rendering

- **Conversion**: `convert_file_obj_to_glb.py` transforms `.obj` files into binary `.glb`.
- **Rendering**: `convert_glb_to_2d_imgs.py` produces 20 uniformly spaced views (18° increments) around each model.

### 2. Vision–Language Embedding

- **Model**: SIGLIP ViT-SO400M-16-SigLIP2-512
- **Embedding**: `embed_2d_imgs.py` encodes each view and stores vectors in Milvus with a FLAT index.

### 3. Caption Generation

- **Model**: Fine-tuned BLIP-2 (Flan-T5-xl)
- **Training**:
  - Epochs: 20   
  - Batch size: 32
  - Learning rate: 2e-5
  - Weight decay: 0.01

### 4. Retrieval Process

1. **Query Embedding**: User text query → SIGLIP → vector
2. **Stage 1 Search**: Milvus cosine similarity → top-k views
3. **Captioning**: BLIP-2 generates captions for top candidates
4. **Re-ranking**: BGE-M3 computes text embedding similarity between query and captions
5. **Output**: Ranked list of 3D object IDs

---

## ⚙️ Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-org/ROOMELSA.git
   cd ROOMELSA
   ```

2. **Setup environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Milvus**:
   - Follow [Milvus installation guide](https://milvus.io/docs/install_standalone.md)
   - Create a collection with FLAT index type

---

## 🎯 Usage

### Data Preparation

```bash
python scripts/convert_file_obj_to_glb.py \
  --root_dir /path/to/obj_files \
  --output_dir /path/to/glb_files

python scripts/convert_glb_to_2d_imgs.py \
  --root_dir /path/to/glb_files \
  --output_dir /path/to/rendered_images
```

### Generating Embeddings

```bash
python scripts/embed_2d_imgs.py \
  --root_dir /path/to/rendered_images \
  --collection_name your_milvus_collection
```

### Training BLIP-2

1. **Make Dataset**:
   ```bash
   python scripts/make_dataset_for_blip.py \
     --src_root /path/to/public_data \
     --dst_root /path/to/training_data
   ```

2. **Train**:
   ```bash
   python scripts/train_blip.py \
     --data_dir /path/to/training_data \
     --output_dir /path/to/output \
     --checkpoint_dir /path/to/checkpoints
   ```

3. **Optional Hyperparams**:
   ```bash
   # Customize batch size, epochs, learning rate
   python scripts/train_blip.py \
     --batch_size 16 \
     --num_epochs 10 \
     --learning_rate 1e-5
   ```

### Inference

```bash
python scripts/infer_with_caption_n_rerank.py \
  --query_path /path/to/queries.json \
  --collection_name your_milvus_collection \
  --output_csv_path /path/to/results.csv \
  --visualize_path /path/to/visualizations \
  --ckpt_path /path/to/best/fine_tuned_blip2
```

For Stage 1 only (no reranking):

```bash
python scripts/infer_without_caption_n_rerank.py \
  --query_path /path/to/queries.json \
  --collection_name your_milvus_collection \
  --output_csv_path /path/to/fast_results.csv
```

---

## 📈 Results
In the final leaderboard of the challenge, our team secured **1st place** out of 18 participants with the following evaluation metrics:

- **R@1 (Recall@1):** 0.94
- **R@5 (Recall@5):** 1.00  
- **R@10 (Recall@10):** 1.00  
- **MRR (Mean Reciprocal Rank):** 0.97 

![Leaderboard image](https://scontent.fdad3-1.fna.fbcdn.net/v/t39.30808-6/492588698_122222014808142763_3949470163467974835_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=127cfc&_nc_eui2=AeHSU2cEN_Xswn5t6b-2pV4-RvbHIm2X9ChG9scibZf0KDcBeCo2uMCRnMd9yg9UW4heDH_N0G5MJOB1FzRfJBY_&_nc_ohc=HB70uKCdRukQ7kNvwHaUF5d&_nc_oc=AdkSvXzcdXXqxL98YFmChY4IZq_LXlb7qK-jNxlYE1y4UP2Jn2o3BqY7dy5j2wE_r-w&_nc_zt=23&_nc_ht=scontent.fdad3-1.fna&_nc_gid=Ooe0gDULObvhI3xKc_ZPog&oh=00_AfF2DsM2m3R94AhKEx24JuWx2sZN2pNAmitED_fSfxZSrw&oe=6813A6A9)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

