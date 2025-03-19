import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import streamlit as st
from threading import Thread

# 모델 및 토크나이저 로드
@st.cache_resource
def load_model():
    model_name = "LGAI-EXAONE/EXAONE-Deep-2.4B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit 애플리케이션 구성
st.set_page_config(
    page_title="EXAONE-Deep-2.4B AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EXAONE-Deep-2.4B AI Assistant")
st.write("EXAONE-Deep-2.4B 모델과 대화해보세요. 아래에 질문을 입력하고 '응답 생성' 버튼을 눌러보세요.")

# 사용자 입력 받기
user_input = st.text_area("질문을 입력하세요:", height=100)

# 응답 생성 함수
def generate_response(prompt):
    messages = [
        {"role": "system", "content": "당신은 LG AI 연구소의 EXAONE-Deep-2.4B 모델입니다. 사용자에게 도움이 되는 답변을 제공하세요."},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    streamer = TextIteratorStreamer(tokenizer)
    thread = Thread(target=model.generate, kwargs=dict(
        input_ids=input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        streamer=streamer
    ))
    thread.start()

    response = ""
    for new_text in streamer:
        response += new_text
        st.write(response)
    thread.join()
    return response

# 버튼 클릭 시 응답 생성
if st.button("응답 생성"):
    if user_input.strip():
        with st.spinner("응답 생성 중..."):
            response = generate_response(user_input)
            st.write("**응답:**")
            st.write(response)
    else:
        st.write("질문을 입력해주세요.")
