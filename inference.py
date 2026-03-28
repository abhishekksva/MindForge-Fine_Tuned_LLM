"""
Inference script for the fine-tuned GPT2 chatbot.
Loads the saved model and generates responses.
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

def load_model(model_path: str = "./fine_tuned_model"):
    """Load fine-tuned model and tokenizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run train.py first to fine-tune the model."
        )

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate a response for a given instruction.
    Uses the same prompt format as training.
    """
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only new tokens (not the prompt)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part
    if "### Response:" in generated:
        response = generated.split("### Response:")[-1].strip()
        # Stop at next instruction if any
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0].strip()
        return response

    return generated


if __name__ == "__main__":
    print("Loading fine-tuned model...")
    model, tokenizer = load_model()

    print("Model ready! Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        answer = generate_response(model, tokenizer, question)
        print(f"Bot: {answer}\n")
