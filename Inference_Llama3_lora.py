from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

mode_path = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_path = "./llama3_lora"

# 讀取 tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, use_fast=False, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.eos_token_id

# 讀取模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# 讀取 lora 權重
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 訓練訓練
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具體具體 lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

prompt = "爸爸再婚，我是不是就有了個新娘？"
# prompt = "只剩一個心臟了還能活嗎？"
# prompt = "樟腦丸是我吃過最難吃的硬糖有奇怪的味道怎麼還有人買"
# prompt = "馬上要上游泳課了，昨天洗的泳褲還沒乾，怎麼辦?"
# prompt = "為什麼沒人說ABCD型的成語？"
messages = [
    {"role": "system", "content": "回答使用**繁體中文**"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9, 
    temperature=0.5, 
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.encode('<|eot_id|>')[1],
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)