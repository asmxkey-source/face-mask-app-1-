import streamlit as st
import cv2
import tempfile
import google.generativeai as genai
import os

# 페이지 설정
st.set_page_config(page_title="AI 얼굴 마스킹 앱", layout="centered")
st.title("🎥 AI 영상 비식별화 서비스")

# 사이드바에서 API 키 입력 받기
api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")

if api_key:
    genai.configure(api_key=api_key)
    
    uploaded_file = st.file_uploader("마스킹할 영상을 업로드하세요 (MP4, MOV)", type=['mp4', 'mov'])

    if uploaded_file is not None:
        # 임시 파일로 영상 저장
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        
        if st.button("얼굴 마스킹 시작"):
            with st.spinner("AI가 얼굴을 분석하고 가리는 중입니다..."):
                # [여기에 Gemini API 호출 및 OpenCV 처리 로직이 들어갑니다]
                # 샘플로 원본 영상을 그대로 보여주는 코드를 넣어두겠습니다.
                # 실제 구현 시에는 AI Studio에서 생성한 상세 로직을 이 부분에 삽입하세요.
                st.success("처리가 완료되었습니다!")
                st.video(tfile.name) # 결과물 출력
                
                with open(tfile.name, "rb") as file:
                    st.download_button("결과 영상 다운로드", file, "masked_video.mp4")
else:
    st.info("왼쪽 사이드바에 Gemini API Key를 입력해 주세요.")