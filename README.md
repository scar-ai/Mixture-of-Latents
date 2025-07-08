# Advanced MoE Transformer with Mixture-of-Latents Attention

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

![alt](https://seeklogo.com/vector-logo/372199/pytorch)

This repository contains the complete implementation of a sophisticated Transformer-based language model, featuring a unique **Mixture-of-Latents Attention (MLA)** mechanism and a **Mixture-of-Experts (MoE)** feed-forward layer. The model is designed for high-performance text generation and is built to scale efficiently using distributed training.

A regular transformer version of this model (single FFN, no routing) beat gpt-2 large in 2h36 when trained on a node of 8 AMD MI300X with ~300M parameters.

This project provides the full codebase, from the architectural backbone and data processing pipelines to single-GPU and distributed training scripts, and a ready-to-use interactive Streamlit application for inference.

## ✨ Key Features

-   **Mixture-of-Latents Attention (MLA):** A novel attention mechanism first introduced in the [Deepseek-V3 paper](https://arxiv.org/pdf/2412.19437) that splits query and key projections into two paths: a content-based path and a rotary-based path. This allows the model to separately process and weigh contextual information and positional information, leading to more nuanced text generation.
-   **Mixture-of-Experts (MoE) Layers:** The feed-forward network in each Transformer block is replaced with a sparse MoE layer. This allows the model to have a very high parameter count while only activating a small subset of "expert" networks for each token, drastically improving training and inference efficiency.
-   **Rotary Position Embeddings (RoPE):** Implements state-of-the-art relative position embeddings, which are embedded into the MLA mechanism.
-   **Distributed Training Ready:** Includes a script (`main_distributed.py`) that leverages PyTorch's `DistributedDataParallel` (DDP) for robust and scalable multi-GPU training (tested on a node of 8 AMD MI300X).
-   **Custom Data Pipeline:** A dedicated data loader (`OpenWebText.py`) for processing the OpenWebText dataset, including on-the-fly tokenization, cleaning, and batching.
-   **Interactive Demo:** A user-friendly Streamlit application (`user.py`) to interact with the trained model, featuring real-time text generation and adjustable sampling parameters.

## 📂 Repository Structure & File Guide

This repository is organized to provide a clear path from understanding the model's architecture to training it and finally using it for inference.

### 1. Model Architecture

-   **`model.py`**: This is the heart of the project. It defines the complete model architecture, including:
    -   `TheTransformer`: The main class that assembles the entire model.
    -   `MultiHeadAttention`: The custom Mixture-of-Latents Attention implementation.
    -   `GatingNetwork` & `TransformerBlock`: The core components for the Mixture-of-Experts (MoE) layers.
    -   `RotaryPositionEncoding`: The implementation for RoPE.

### 2. Training the Model

The repository includes two scripts for training the model, catering to different hardware setups.

-   **`training.py` (Single-GPU Training)**
    -   **Purpose:** A straightforward script for training the model on a single GPU.
    -   **Details:** It handles data loading, model initialization, a standard training loop with mixed-precision support (`torch.amp`), and a custom learning rate scheduler.
    -   **Use Case:** Ideal for debugging, running smaller-scale experiments, or for users who do not have a multi-GPU environment.

-   **`main_distributed.py` (Multi-GPU Distributed Training)**
    -   **Purpose:** The primary script for training the full-scale model efficiently across multiple GPUs.
    -   **Details:** It leverages PyTorch's `DistributedDataParallel` (DDP) and `DistributedSampler` to parallelize the training process. It also includes an optional token-dropping feature as a regularization technique.
    -   **Use Case:** The recommended script for training the model from scratch to achieve the best performance on large datasets.

### 3. Using the Model for Inference

-   **`user.py` (Interactive Streamlit Demo)**
    -   **Purpose:** A web-based application for generating text with the trained model.
    -   **How to Use:**
        1.  Ensure you have a trained model checkpoint (e.g., `weights/mol.pth`). The script is pre-configured to look for this file.
        2.  Install the required Python packages: `pip install -r requirements.txt`.
        3.  Run the application from your terminal:
            ```bash
            streamlit run user.py
            ```
