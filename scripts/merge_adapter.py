import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
base_model_path = "models/llama3-2-3b-instruct"
adapter_path = "dpop_out"
output_path = "models/llama3-2-3b-dpop-instruct"

print(f"Loading base model from {base_model_path}...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Loading adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging and unloading...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)
print("Done.")