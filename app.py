# import subprocess
# import sys
# Install torch if not already installed
# subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

########################

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 로딩 (캐싱 활용)
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit 앱 제목 설정
st.title("EXAONE 3.0 Demo")

# 프롬프트 입력 텍스트 영역
prompt = st.text_area("프롬프트를 입력하세요:", value="너의 소원을 말해봐")

# 실행 버튼
if st.button("실행"):
    # 메시지 생성
    messages = [
        {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # 모델 실행 및 결과 출력
    with st.spinner("생성 중..."):
        output = model.generate(
            input_ids.to("cuda"),  # Streamlit Cloud는 CPU만 지원하므로 이 부분 수정 필요
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128
        )
    st.write(tokenizer.decode(output[0]))
