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

📈 Results

The table below shows the training and validation performance per epoch for one of the fine-tuned models (likely roberta-base). 

![Training performance for Roberta](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/after_training.png)

Final model performance on validation set (Epoch 9):

![Performance on valid set for Roberta](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/for_valid_metr.png)

![Confusion Matrix](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/for_valid.png)

Final model performance on test set (Epoch 9):

![Performance on test set for Roberta](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/for_test_metr.png)

These metrics demonstrate stable convergence and high performance on arXiv abstract classification, confirming that the model generalizes well to unseen data within the same distribution.

The table below shows the training and validation performance per epoch for scibert-scivocab-uncased + LoRA fine-tuning. 

![Training performance for Lora](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/after_training_lora.png)

Final model performance on validation set (Epoch 10):

![Performance on valid set for Lora](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/for_valid_lora.png)

![Confusion Matrix_lora](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/matrix_lora.png)

Final model performance on test set (Epoch 10):

![Performance on test set for Roberta](https://github.com/yuliadziuba/arXiv-Scientific-Paper-Classification/blob/main/for_test_lora.png)

🤖 Model Comparison: RoBERTa vs SciBERT + LoRA
We compared two transformer-based models for the arXiv abstract classification task:

RoBERTa-base: fully fine-tuned

allenai/scibert_scivocab_uncased: fine-tuned using LoRA (Low-Rank Adaptation) for parameter-efficient tuning

Model	Accuracy ↑	F1 Score ↑	Precision ↑	Recall ↑	Params Updated

RoBERTa-base	86.68%	86.64%	86.75%	86.68%	100% (full)

SciBERT + LoRA	86.20%	86.08%	86.12%	86.20%	~0.5% (LoRA)

📝 Summary
RoBERTa-base achieved slightly higher performance across all metrics.

SciBERT + LoRA performed competitively while updating only a small fraction of the model's parameters, making it an ideal choice for resource-constrained environments.

Both models demonstrated strong generalization on the validation set.

📌 For full training logs and evaluation metrics, refer to the respective notebooks included in this repository.


