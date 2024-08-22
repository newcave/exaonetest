import transformers
import streamlit as st

# LLaMA 2 7B Chat 모델 로드
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Streamlit 앱
st.title("LLaMA AI Assistant")
st.write("LLaMA 2 7B Chat Model")

# 프롬프트 입력
prompt = st.text_input("Enter your prompt", value="Explain who you are")

# 버튼 클릭 시 응답 생성
if st.button("Generate Response"):
    with st.spinner("Generating..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("**Response:**")
        st.write(response)
