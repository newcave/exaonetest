import streamlit as st
import requests

# Hugging Face 모델 정보
MODEL_NAME = "LGAI-EXAONE/EXAONE-Deep-2.4B"
API_URL = f"https://api-inference.huggingface.co/models/EXAONE-Deep-2.4B"

# Streamlit 시크릿 매니저에서 Hugging Face 액세스 토큰 가져오기
hf_token = st.secrets["huggingface"]["token"]
headers = {"Authorization": f"Bearer {hf_token}"}

st.title("EXAONE-Deep-2.4B 텍스트 생성기")

# 사용자 입력 받기
user_input = st.text_area("텍스트를 입력하세요:")

if st.button("생성"):
    if user_input:
        # API에 요청 보낼 데이터 구성
        payload = {"inputs": user_input}
        try:
            # API 요청 보내기
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            # API 응답 받기
            result = response.json()
            generated_text = result[0].get("generated_text", "응답에서 텍스트를 찾을 수 없습니다.")
        except requests.exceptions.RequestException as e:
            generated_text = f"API 요청 중 오류 발생: {e}"
        except (KeyError, IndexError) as e:
            generated_text = f"응답 처리 중 오류 발생: {e}"

        st.write("생성된 텍스트:")
        st.write(generated_text)
    else:
        st.write("텍스트를 입력해주세요.")
