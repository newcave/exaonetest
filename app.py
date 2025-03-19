import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# Hugging Face 토큰 설정
hf_token = st.secrets["huggingface"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# 모델과 토크나이저 로드 함수
@st.cache_resource
def load_model():
    model_name = "LGAI-EXAONE/EXAONE-Deep-2.4B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map='auto'  # 가용한 장치에 자동으로 할당
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# 모델과 토크나이저 로드
model, tokenizer = load_model()

# Streamlit 인터페이스 구성
st.title("EXAONE-Deep-2.4B 텍스트 생성기")

# 사용자 입력 받기
user_input = st.text_area("텍스트를 입력하세요:", "")

# 입력이 있을 경우 출력 생성
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("생성된 텍스트:")
    st.write(generated_text)
