# arXiv-Scientific-Paper-Classification

This project focuses on fine-tuning transformer-based models for multi-class classification of scientific papers from the arXiv dataset provided by Hugging Face (ccdv/arxiv-classification). The goal is to predict the paper's subject area (one of 11 categories) based on its title and abstract.

📦 Dataset
We use the ccdv/arxiv-classification dataset, which contains:

Scientific paper entries (title + abstract)

11 top-level arXiv categories as classification labels
(e.g. cs.AI, math.PR, stat.ML, etc.)

Pre-split into train, validation, and test sets

📁 Notebooks
The repository contains two notebooks:

📘 1. roberta-base fine-tuning
Model: roberta-base

Fine-tuned using Hugging Face's Trainer API

Tokenized with max length 512

Evaluated using:

Accuracy

Precision / Recall / F1 (weighted)

Confusion Matrix

🧪 2. scibert-scivocab-uncased + LoRA fine-tuning
Model: allenai/scibert_scivocab_uncased

Lightweight fine-tuning using LoRA (Low-Rank Adaptation) via PEFT

Same dataset splits and hyperparameters as in roberta-base

📊 Evaluation
Evaluated with the same metrics:

Accuracy

Precision / Recall / F1

Confusion Matrix

For both models, the following metrics are logged and visualized:

A comparative analysis is included in each notebook to evaluate model performance on the test set.

✨ Goal
This project demonstrates:

The effectiveness of domain-specific models like SciBERT for scientific NLP

The benefits of using parameter-efficient fine-tuning (LoRA) to reduce computational cost while maintaining high accuracy

A clean and reproducible workflow for multi-class text classification with Hugging Face Transformers and Datasets
