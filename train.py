"""
Fine-tune GPT-2 on custom Q&A data using HuggingFace Transformers.
This script handles: data prep → tokenization → training → saving model
"""

from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json
import os

# ── 1. Sample Training Data ───────────────────────────────────────────────────
# Format: Instruction → Response pairs (like a Q&A assistant)
TRAINING_DATA = [
    {"instruction": "What is machine learning?",
     "response": "Machine learning is a subset of AI where models learn patterns from data without being explicitly programmed."},
    {"instruction": "What is deep learning?",
     "response": "Deep learning uses neural networks with many layers to learn complex patterns from large datasets."},
    {"instruction": "What is a neural network?",
     "response": "A neural network is a series of algorithms that recognizes patterns by mimicking the human brain structure."},
    {"instruction": "What is NLP?",
     "response": "Natural Language Processing (NLP) is a branch of AI that enables machines to understand and generate human language."},
    {"instruction": "What is overfitting?",
     "response": "Overfitting occurs when a model learns training data too well, including noise, causing poor performance on new data."},
    {"instruction": "What is a transformer model?",
     "response": "A transformer is a deep learning architecture using self-attention mechanisms, forming the basis of models like BERT and GPT."},
    {"instruction": "What is fine-tuning?",
     "response": "Fine-tuning is the process of taking a pre-trained model and training it further on a specific dataset for a targeted task."},
    {"instruction": "What is RAG?",
     "response": "RAG stands for Retrieval-Augmented Generation — it combines document retrieval with LLM generation for grounded answers."},
    {"instruction": "What is FAISS?",
     "response": "FAISS is Facebook's library for efficient similarity search and clustering of dense vectors at scale."},
    {"instruction": "What is the difference between supervised and unsupervised learning?",
     "response": "Supervised learning uses labeled data to train models, while unsupervised learning finds hidden patterns in unlabeled data."},
    {"instruction": "What is gradient descent?",
     "response": "Gradient descent is an optimization algorithm that minimizes the loss function by iteratively updating model parameters."},
    {"instruction": "What is a loss function?",
     "response": "A loss function measures how far a model's predictions are from the actual values, guiding the training process."},
    {"instruction": "What is backpropagation?",
     "response": "Backpropagation computes gradients of the loss with respect to model weights by propagating errors backward through the network."},
    {"instruction": "What is transfer learning?",
     "response": "Transfer learning reuses a model trained on one task as a starting point for a different but related task."},
    {"instruction": "What is tokenization?",
     "response": "Tokenization splits text into smaller units called tokens — words, subwords, or characters — that models can process."},
]

def prepare_dataset(data: list) -> Dataset:
    """Convert Q&A pairs to instruction-following format."""
    texts = []
    for item in data:
        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}\n\n"
        texts.append({"text": text})
    return Dataset.from_list(texts)


def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize text examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )


def fine_tune_model(
    model_name: str = "gpt2",
    output_dir: str = "./fine_tuned_model",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5
):
    """
    Main fine-tuning function.
    """
    print(f"Loading tokenizer and model: {model_name}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token

    # ── Load model ────────────────────────────────────────────────────────────
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # ── Prepare dataset ───────────────────────────────────────────────────────
    print("Preparing dataset...")
    dataset = prepare_dataset(TRAINING_DATA)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none"
    )

    # ── Data collator ─────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT2 is causal LM, not masked
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting fine-tuning...")
    trainer.train()

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete!")

    return output_dir


if __name__ == "__main__":
    fine_tune_model()
