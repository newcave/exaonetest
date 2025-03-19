import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import streamlit as st
from threading import Thread

# Hugging Face 토큰 설정
hf_token = st.secrets["huggingface"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

@st.cache_resource
def load_model():
    model_name = "LGAI-EXAONE/EXAONE-Deep-2.4B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'  # 가용한 GPU에 모델 로드
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

def generate_response(prompt, streaming=True):
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    if streaming:
        streamer = TextIteratorStreamer(tokenizer)
        thread = Thread(target=model.generate, kwargs=dict(
            input_ids=input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=32768,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            streamer=streamer
        ))
        thread.start()

        response = ""
        for text in streamer:
            response += text
            st.write(text, end="", flush=True)
        return response
    else:
        output = model.generate(
            input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=32768,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )
        return tokenizer.decode(output[0])

st.title("EXAONE-Deep-2.4B Chatbot")

user_input = st.text_input("사용자 입력:")

if st.button("전송"):
    if user_input:
        st.write("답변 생성 중...")
        response = generate_response(user_input)
        st.write("AI:", response)
    else:
        st.write("입력을 제공해주세요.")
