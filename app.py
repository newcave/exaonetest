import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import os

@st.cache_resource
def load_model():
    model_name = "LGAI-EXAONE/EXAONE-Deep-2.4B"
    
    # Hugging Face API 토큰 설정
    hf_token = st.secrets["huggingface"]["token"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map='cpu'  # CPU에서 모델 로드
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit 인터페이스 구성
st.title("Hugging Face 모델 테스트")
user_input = st.text_area("텍스트를 입력하세요:")

if st.button("생성"):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("생성된 텍스트:")
    st.write(result)
