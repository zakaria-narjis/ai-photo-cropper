# AI Photo Cropper

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

An AI-powered photo cropping web application offering two intelligent cropping modes: **composition-aware cropping** using a custom VGG16-based neural network, and **text-query cropping** using OpenAI CLIP and YOLOv5.

---

## Features

- **Composition-aware cropping** — a CACNet model trained on the FLMS dataset predicts the aesthetically optimal bounding box for any photo
- **Batch processing** — upload multiple images at once and download all crops as a zip archive
- **Text-query cropping** — describe what to keep in plain English (e.g. `"cat"`, `"company logo"`) and CLIP finds the best-matching detected region
- **Manual crop fallback** — drag-to-crop UI via `streamlit-cropper` when AI is not needed
- **GPU-accelerated inference** — full CUDA support with automatic CPU fallback
- **Interactive API docs** — Swagger UI available at `/docs` out of the box
- **One-command setup** — fully containerised with Docker Compose

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        Docker Compose                        │
│                                                              │
│  ┌───────────────────────┐      ┌────────────────────────┐   │
│  │      Frontend          │      │       Backend          │   │
│  │   Streamlit  :8501     │─────▶│    FastAPI   :8080     │   │
│  │                        │      │                        │   │
│  │  Home                  │      │  GET  /                │   │
│  │  Composition Cropping  │      │  POST /one_crop/       │   │
│  │  Crop By Query         │      │  POST /multi_crop/     │   │
│  └───────────────────────┘      │  POST /clip_crop/      │   │
│                                  │                        │   │
│                                  │  ┌──────────────────┐  │   │
│                                  │  │  CACNet (VGG16)  │  │   │
│                                  │  │  Composition     │  │   │
│                                  │  │  Cropping        │  │   │
│                                  │  └──────────────────┘  │   │
│                                  │  ┌──────────────────┐  │   │
│                                  │  │  YOLOv5 + CLIP   │  │   │
│                                  │  │  Query Cropping  │  │   │
│                                  │  └──────────────────┘  │   │
│                                  └────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Frontend | Streamlit 1.37 | Interactive web UI |
| Backend | FastAPI 0.111 + Uvicorn | Async REST API |
| Composition model | CACNet (VGG16) | Aesthetic crop prediction |
| Object detection | YOLOv5s | Region proposals for query cropping |
| Semantic matching | OpenAI CLIP ViT-B/32 | Text-to-image similarity scoring |
| Deep learning runtime | PyTorch 2.x + CUDA 12.1 | GPU-accelerated inference |
| Containerisation | Docker Compose | Service orchestration |
| Linting & formatting | Ruff | PEP 8 compliance |

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) *(optional — CPU fallback is automatic)*

### Run

```bash
git clone https://github.com/zakaria-narjis/ai-photo-cropper.git
cd ai-photo-cropper
docker compose up --build
```

| Service | URL |
| --- | --- |
| Frontend | <http://localhost:8501> |
| Backend API docs | <http://localhost:8080/docs> |

---

## How It Works

### Mode 1 — Composition-Aware Cropping

Uses **CACNet** (Composition-Aware Cropping Network), a VGG16-based encoder enhanced with a Key-Content Module (KCM). The model is trained on the [FLMS dataset](http://fangchen.org/proj_page/FLMS_mm14/FLMS_mm14.html) and jointly:

1. Classifies the dominant photographic composition rule (rule of thirds, symmetry, diagonal, etc.) using Class Activation Maps.
2. Regresses a bounding box that maximises the aesthetic score for that composition type.

Images are resized to 224 × 224 for inference; the predicted crop coordinates are rescaled back to the original resolution before being returned.

### Mode 2 — Text-Query Cropping

Combines **YOLOv5s** for region proposals with **OpenAI CLIP** (ViT-B/32) for semantic matching:

1. YOLOv5 detects all objects in the uploaded image.
2. Each detected region is encoded into a CLIP image embedding.
3. The user's text query is encoded into a CLIP text embedding.
4. Cosine similarity is computed between the text embedding and all region embeddings.
5. The region with the highest similarity score is returned as the crop.

---

## API Reference

| Method | Endpoint | Input | Output |
| --- | --- | --- | --- |
| `GET` | `/` | — | `{"health_check": "OK"}` |
| `POST` | `/one_crop/` | `image` (file) | `{x1, y1, x2, y2}` |
| `POST` | `/multi_crop/` | `images[]` (files) | `{crops: [{image_name, coords}]}` |
| `POST` | `/clip_crop/` | `image` (file), `query` (str) | `{x1, y1, x2, y2}` |

All coordinates are pixel values relative to the original uploaded image dimensions. Full interactive documentation is available at <http://localhost:8080/docs> when the backend is running.

---

## Project Structure

```text
ai-photo-cropper/
├── docker-compose.yml
├── ruff.toml
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api.py                          # FastAPI app and route handlers
│   ├── clipcrop/
│   │   └── clipcrop.py                 # YOLOv5 + CLIP crop pipeline
│   └── comp_cropping/
│       ├── CACNet.py                   # VGG16-based composition model definition
│       ├── crop.py                     # Cropper class + DataLoader pipeline
│       └── pretrained_models/
│           └── best-FLMS_iou.pth       # CACNet pretrained weights
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── Home.py                         # Streamlit entry point
│   ├── api_handler.py                  # HTTP client wrappers for the backend API
│   ├── user_session.py                 # Session state model
│   ├── utils.py                        # Image utility functions
│   ├── pages/
│   │   ├── Composition Cropping.py     # Composition crop UI page
│   │   └── Crop_By_Query.py            # CLIP crop UI page
│   └── sample_images/                  # Bundled demo images (Lübeck cityscape)
└── sample_images/                      # Original / cropped result pairs
    ├── original/
    └── cropped/
```

---

## Sample Images

The `frontend/sample_images/` directory includes four Lübeck cityscape photographs bundled as built-in demos. Select any of them from the dropdown on the **Composition Cropping** page to try the app without uploading your own image.
