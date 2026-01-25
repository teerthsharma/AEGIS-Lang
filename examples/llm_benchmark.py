
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain quantum physics to a 5 year old."
MAX_TOKENS = 50

def run_benchmark():
    print(f"Benchmarking Python (Transformers) - {MODEL_ID}")
    
    # Measure Load Time
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    end_load = time.time()
    
    print(f"Load Time: {end_load - start_load:.4f} seconds")
    
    # Prepare Input
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    
    # Measure Inference Time
    print(f"Generating (Prompt: '{PROMPT}')...")
    start_gen = time.time()
    outputs = model.generate(input_ids, max_new_tokens=MAX_TOKENS)
    end_gen = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    gen_time = end_gen - start_gen
    tokens_per_sec = MAX_TOKENS / gen_time
    
    print("-" * 40)
    print(f"Generated: {generated_text[:100]}...")
    print("-" * 40)
    print(f"Inference Time: {gen_time:.4f} seconds")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark()
