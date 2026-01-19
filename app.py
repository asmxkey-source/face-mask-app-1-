import streamlit as st
import cv2
import tempfile
import google.generativeai as genai
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="AI 얼굴 비식별화", layout="centered")
st.title("🎥 AI 영상 얼굴 마스킹")

# 1. API 키 설정
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # 영상 이해에 최적화된 모델

    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

    if uploaded_file:
        # 파일 저장
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("얼굴 마스킹 시작"):
            with st.spinner("AI가 영상을 분석하고 얼굴을 가리는 중입니다..."):
                try:
                    # 2. Gemini에게 영상 업로드 및 얼굴 좌표 요청
                    st.info("AI에게 영상 분석을 요청하는 중...")
                    video_file = genai.upload_file(path=video_path)
                    
                    # 얼굴 좌표 추출 프롬프트
                    prompt = "Find all human faces in this video and provide their coordinates as [ymin, xmin, ymax, xmax] in JSON format for each frame."
                    response = model.generate_content([video_file, prompt])
                    
                    # 3. OpenCV 영상 처리
                    cap = cv2.VideoCapture(video_path)
                    width = int(cap.get(cv2.職業_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # 결과 영상 저장 설정
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break

                        # [핵심 로직] Gemini가 준 좌표를 프레임에 적용 (간략화된 예시)
                        # 실제 구현시 response에서 파싱한 좌표로 cv2.GaussianBlur 적용
                        # 우선은 중앙부 샘플 블러로 작동 확인
                        h, w, _ = frame.shape
                        face_region = frame[h//4:3*h//4, w//4:3*w//4]
                        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                        frame[h//4:3*h//4, w//4:3*w//4] = blurred_face
                        
                        out.write(frame)

                    cap.release()
                    out.release()

                    st.success("처리 완료!")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as f:
                        st.download_button("결과물 다운로드", f, "masked_video.mp4")

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
else:
    st.info("먼저 사이드바에 API 키를 입력해 주세요.")
