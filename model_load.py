from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Tokenizer와 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("mlfoundations/tabula-8b")
model = AutoModelForCausalLM.from_pretrained("mlfoundations/tabula-8b")

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 입력 예제
input_text = "What is the defect prediction rate for"
inputs = tokenizer(input_text, return_tensors="pt").to(device)  # 입력도 GPU로 이동

# 모델 추론(Inference)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
