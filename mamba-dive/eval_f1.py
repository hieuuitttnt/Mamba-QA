from sklearn.metrics import precision_recall_fscore_support
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import sys
import json
import pandas as pd
import time
from tqdm import tqdm
from transformers import AutoModel

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Calculate the F1 score between the prediction and ground truth.
    Tokenize the input and calculate overlap.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def run_mamba(model, question, context):
    text = f"{context}\n\nQ: {question}\nA:"
    input_ids = torch.LongTensor([tokenizer.encode(text)]).cuda()
    
    out = model.generate(
        input_ids=input_ids,
        max_length=len(input_ids)+128,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.batch_decode(out)[0]
    cleaned = decoded.replace(text, "").replace("<|endoftext|>", "")
    answer = cleaned.split("\n\n")[0].strip()
    return answer


def write_results(results, output_file):
    df = pd.DataFrame(results)
    df = df[["idx", "context", "question", "answer", "guess", "f1", "time"]]
    print(f"Writing {output_file}")
    df.to_json(output_file, orient="records", lines=True)


# Load model and tokenizer
model = sys.argv[1]
dataset = sys.argv[2]
output_file = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

model = MambaLMHeadModel.from_pretrained(model, device="cuda", dtype=torch.float16)

results = []
with open(dataset) as f:
    all_data = [json.loads(line) for line in tqdm(f)]
    
    total_qs = len(all_data)

    for i, data in enumerate(all_data):
        start_time = time.time()

        question = data["prompt"]
        context = data["context"]
        answer = data["response"]
        guess = run_mamba(model, question, context)
        end_time = time.time()

        # Calculate F1 score
        f1 = f1_score(guess, answer)

        print(f"Question {i}/{total_qs}")
        print(f"Context: {context}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"?: {guess}")
        print(f"F1 Score: {f1:.4f}")
        print("=" * 80)
        
        result = {
            "idx": i,
            "question": question,
            "context": context,
            "answer": answer,
            "guess": guess,
            "f1": f1,
            "time": end_time - start_time
        }
        results.append(result)

        if len(results) % 20 == 0:
            write_results(results, output_file)
            
write_results(results, output_file)
