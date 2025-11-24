from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "allenai/truthfulqa-truth-judge-llama2-7B"
local_dir = "/home/rs_students/nlpG4/deepseek_aligner/models/truthfulqa-truth-judge-llama2-7B"

# Download and save to 'models/' for first use
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)
