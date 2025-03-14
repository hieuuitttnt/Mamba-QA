import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import sys
import json
import pandas as pd
import time
from tqdm import tqdm
from transformers import AutoModel
def run_mamba(model, question, context):

    text = f"{context}\n\nQ: {question}\nA:"
    # print(text)
    input_ids = torch.LongTensor([tokenizer.encode(text)]).cuda()
    # print(input_ids)
    
    out = model.generate(
        input_ids=input_ids,
        max_length=len(input_ids)+128,
        eos_token_id=tokenizer.eos_token_id
    )

    # print(out)
    decoded = tokenizer.batch_decode(out)[0]
    # print("="*80)
    # print(decoded)
    
    # out returns the whole sequence plus the original
    cleaned = decoded.replace(text, "")
    cleaned = cleaned.replace("<|endoftext|>", "")
    
    # the model will just keep generating, so only grab the first one
    answer = cleaned.split("\n\n")[0].strip()
    # print(answer)
    return answer

def write_results(results, output_file):
    df = pd.DataFrame(results)
    df = df[["idx", "context", "question", "answer", "guess", "is_correct", "time"]]

    print(f"Writing {output_file}")
    df.to_json(output_file, orient="records", lines=True)

model = sys.argv[1]
# model = "Oxen-AI/mamba-130m-context"
dataset = sys.argv[2]
output_file = sys.argv[3]
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

model = MambaLMHeadModel.from_pretrained(model, device="cuda", dtype=torch.float16)

results = []
with open(dataset) as f:
    all_data = []
    for line in tqdm(f):
        data = json.loads(line)
        all_data.append(data)

    total_qs = len(all_data)
    for i, data in enumerate(all_data):
        start_time = time.time()

        # print(data)
        question = data["prompt"]
        context = data["context"]
        answer = data["response"]
        guess = run_mamba(model, question, context)
        end_time = time.time()
        is_correct = (answer.strip().lower() == guess.strip().lower())
        print(f"Question {i}/{total_qs}")
        print(f"Context: {context}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"?: {guess}")
        if is_correct:
            print(f"✅")
        else:
            print(f"❌")
        print("="*80)
        result = {
            "idx": i,
            "question": question,
            "context": context,
            "answer": answer,
            "guess": guess,
            "is_correct": is_correct,
            "time": end_time - start_time
        }
        results.append(result)

        if len(results) % 20 == 0:
            write_results(results, output_file)
            
        # if len(results) > 100:
        #     break

write_results(results, output_file)
