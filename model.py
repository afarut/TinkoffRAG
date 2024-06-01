from config import TOKEN, DEFAULT_SYSTEM_PROMPT
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class RAG:
    def __init__(self):
        MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            force_download=False,
            token=TOKEN
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
        self.generation_config = GenerationConfig.from_pretrained(MODEL_NAME, token=TOKEN)


    def generate(self, query):
        prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": query
        }], tokenize=False, add_generation_prompt=True)
        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(**data, generation_config=self.generation_config)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return output