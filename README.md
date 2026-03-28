# MindForge — Fine-tuned LLM Chatbot ⚡

## Overview
A custom AI chatbot built by fine-tuning GPT-2 on domain-specific Q&A data using HuggingFace Transformers. Demonstrates full fine-tuning pipeline from data preparation to deployment.

## Architecture
```
Custom Q&A Data → Tokenization → GPT-2 Fine-tuning → Saved Model → Streamlit Inference
```

## Tech Stack
- **HuggingFace Transformers** — Model loading, training, inference
- **GPT-2** — Base pre-trained language model
- **HuggingFace Datasets** — Dataset preparation
- **PyTorch** — Training backend
- **Streamlit** — Interactive chat UI

## Features
- Full fine-tuning pipeline on custom instruction-response data
- Instruction-following format (### Instruction / ### Response)
- Adjustable temperature and token length at inference
- Clean chat interface with history
- Modular train/inference separation

## Project Structure
```
FineTuned_LLM_Chatbot/
├── train.py          ← Fine-tuning pipeline
├── inference.py      ← Model loading & generation
├── app.py            ← Streamlit chat UI
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-tune the model
```bash
python train.py
```
This saves the model to `./fine_tuned_model/`

### 3. Launch the app
```bash
streamlit run app.py
```

## Fine-tuning Details
- Base model: GPT-2 (124M parameters)
- Training epochs: 3
- Batch size: 4
- Learning rate: 5e-5
- Data format: Instruction-Response pairs
- Domain: AI & Machine Learning Q&A

## Resume Bullet Points
- Fine-tuned GPT-2 on custom Q&A dataset using HuggingFace Transformers with instruction-following format
- Implemented complete pipeline: data prep → tokenization → Trainer API → model saving → inference
- Deployed interactive chatbot on Streamlit with adjustable generation parameters

## Live Demo
[Streamlit App Link]
