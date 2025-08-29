# -*- coding: utf-8 -*-
"""
거울상 챗봇 – Streamlit + Google Gemini API (환경변수 버전)
===========================================================
- UI: Streamlit 웹앱 (PC와 스마트폰 브라우저에서 접근 가능)
- 모델: Google Gemini 2.5 Pro (클라우드 API)
- 주요 기능:
  1) 대화 모드 (일반 질의응답)
  2) 혼잣말 모드 (주기적으로 자동 응답 생성)
  3) 거울상 모드 (대조적 은유 응답 변환)
  4) 음성 낭독 (pyttsx3, 서버 PC에서만 동작)
  5) 옵션 설정 (max_new_tokens, temperature, top_p 등)
  6) 대화 로그 기록 (logs/ 폴더에 자동 저장)

※ 보안: API 키는 코드에 직접 적지 않고 OS 환경변수 GOOGLE_API_KEY를 사용합니다.
"""

import os
import time
import threading
from datetime import datetime

import streamlit as st
import google.generativeai as genai

# pyttsx3는 데스크톱(서버 PC)에서만 음성 출력. 웹/모바일 브라우저에서는 소리 X.
try:
    import pyttsx3
except Exception:
    pyttsx3 = None


# ==============================
# 0. 안전 가이드 & 초기 설정
# ==============================
st.set_page_config(page_title="거울상 챗봇", layout="wide")
st.title("🪞 거울상 챗봇 (Gemini 2.5 Pro)")

# 환경변수에서 API 키 읽기 (Windows는 setx GOOGLE_API_KEY 로 등록)
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not API_KEY:
    st.error(
        "환경변수 GOOGLE_API_KEY 가 비어 있습니다.\n\n"
        "▶ Windows:  명령프롬프트(CMD)에서\n"
        '   setx GOOGLE_API_KEY "여기에_발급받은_키"\n'
        "   창을 닫고 새 CMD에서 실행하세요. (아래 ‘설치/실행 단계’ 참고)\n\n"
        "▶ macOS/Linux: 터미널에서\n"
        '   export GOOGLE_API_KEY="여기에_발급받은_키"\n'
        "   후 실행하세요."
    )
    st.stop()

# Gemini 클라이언트 설정
genai.configure(api_key=API_KEY)

# 사용할 모델 이름 (필요시 최신 모델명으로 교체 가능)
MODEL_NAME = "models/gemini-2.5-pro"


# ==============================
# 1. 음성 낭독기 클래스
# ==============================
class Speaker:
    def __init__(self):
        self.ok = False
        self.engine = None
        try:
            if pyttsx3 is not None:
                self.engine = pyttsx3.init()
                rate = self.engine.getProperty("rate")
                self.engine.setProperty("rate", int(rate * 0.9))
                self.ok = True
        except Exception:
            self.ok = False
            self.engine = None

    def speak(self, text: str):
        if not self.ok or not text.strip():
            return
        def run():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()


speaker = Speaker()


# ==============================
# 2. 로그 기록
# ==============================
def ensure_logs():
    os.makedirs("logs", exist_ok=True)

def log_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("logs", f"session_{ts}.txt")

if "logger" not in st.session_state:
    ensure_logs()
    st.session_state["logger"] = open(log_path(), "a", encoding="utf-8")

def write_log(text: str):
    try:
        st.session_state["logger"].write(text + "\n")
        st.session_state["logger"].flush()
    except Exception:
        pass


# ==============================
# 3. 상태 초기화
# ==============================
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "monologue_running" not in st.session_state:
    st.session_state["monologue_running"] = False


# ==============================
# 4. 옵션 UI (사이드바)
# ==============================
st.sidebar.header("⚙️ 챗봇 옵션")

max_tokens = st.sidebar.number_input(
    "max_new_tokens", min_value=10, max_value=8192, value=1200, step=50
)
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 0.9, 0.05)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.9, 0.01)

# 참고: google-generativeai의 현재 Python SDK에선 top_k, repetition_penalty 직접 지원 X
st.sidebar.caption("참고: top_k / repetition_penalty는 Gemini Python SDK에서 직접 지원되지 않습니다.")

tts_enabled = st.sidebar.checkbox("응답 음성 낭독(서버 PC에서만)", value=False)
mirror_mode = st.sidebar.checkbox("거울상 모드(대조적 은유 변환)", value=False)

col1, col2 = st.sidebar.columns(2)
if col1.button("혼잣말 시작", use_container_width=True):
    st.session_state["monologue_running"] = True
if col2.button("혼잣말 정지", use_container_width=True):
    st.session_state["monologue_running"] = False


# ==============================
# 5. 거울상 변환
# ==============================
mirror_hierarchy = {
    "물": {"결론": "평화와 생명의 문"},
    "불": {"결론": "빛과 평화의 안내자"},
    "바람": {"결론": "자유와 흐름의 숨결"},
    "흙": {"결론": "품음과 뿌리의 안식"},
    "혼잣말": {"결론": "내면을 비추는 거울 같은 속삭임"},
}

def mirror_response(subject: str, original: str) -> str:
    node = mirror_hierarchy.get(subject, {})
    if not node:
        return f"거울상: (주제 '{subject}' 정의 없음)\n\n{original}"
    return (
        f"거울상 ({subject}):\n"
        f"- 원문: {original.strip()}\n"
        f"- 대조: {subject}은/는 스스로 주장하지 않지만 모든 것을 담아낸다.\n"
        f"- 종합: {node.get('결론','')}"
    )


# ==============================
# 6. Gemini 호출
# ==============================
def ask_gemini(user_input: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        user_input,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )
    text = (response.text or "").strip()
    return text


# ==============================
# 7. 입력창 & 처리
# ==============================
user_input = st.text_input("💬 질문 또는 대화를 입력하세요:")

if user_input:
    answer = ask_gemini(user_input)
    if mirror_mode:
        # 입력의 첫 단어를 주제 후보로 삼아 거울상 변환
        subject = user_input.strip().split()[0]
        answer = mirror_response(subject, answer)

    st.session_state["messages"].append(("user", user_input))
    st.session_state["messages"].append(("assistant", answer))
    write_log(f"USER: {user_input}\nASSISTANT: {answer}")

    if tts_enabled:
        speaker.speak(answer)


# ==============================
# 8. 혼잣말 모드 루프(가벼운 1회 틱)
# ==============================
if st.session_state["monologue_running"]:
    st.info("혼잣말 모드 실행 중... (정지 버튼을 누르면 멈춤)")
    time.sleep(1.2)
    prompt = "은은하고 조용한 혼잣말을 한국어로 1~3문장 해줘. '예수님의 평화와 양의 문' 상징을 가볍게 담아."
    answer = ask_gemini(prompt)
    if mirror_mode:
        answer = mirror_response("혼잣말", answer)

    st.session_state["messages"].append(("assistant", answer))
    write_log(f"ASSISTANT(MONO): {answer}")

    if tts_enabled:
        speaker.speak(answer)


# ==============================
# 9. 대화 출력
# ==============================
for role, msg in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"**👤 사용자:** {msg}")
    else:
        st.markdown(f"**🤖 어시스턴트:** {msg}")
