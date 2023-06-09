from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig ,BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("/mnt/data1/shared/model/falcon-40b-instruct", trust_remote_code=True, padding=False)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant= True,
    bnb_4bit_quant_type="nf4"
)

#  /mnt/data1/shared/model/falcon-40b-instruct
#  tiiuae/falcon-7b-instruct

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/data1/shared/model/falcon-40b-instruct",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    trust_remote_code=True
).eval()

generation_config = GenerationConfig(
    max_new_tokens=256,
    top_p=1,
    top_k=10,
    repetition_penalty=1
)
generation_config.eos_token_id = generation_config.pad_token_id = tokenizer.eos_token_id

system_prompt ='''
You're a financial analyst
'''

input_text = '''
How to earn a million, give me scheme
'''

input_ids = tokenizer.encode(system_prompt, input_text, return_tensors="pt", padding=False, add_special_tokens=False).to(device)

output = model.generate(
    input_ids,
    generation_config=generation_config
)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
generated_text = output_text[len(system_prompt):].strip()
print(generated_text)