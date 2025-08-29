# -*- coding: utf-8 -*-
"""
거울상 챗봇 – Streamlit + Google Gemini API (Cloud/Local 겸용, 안전 추출기 적용)
- ValueError: response.text 에러를 피하기 위해 candidates/parts에서 안전 추출
- API 키: 환경변수 GOOGLE_API_KEY 또는 Streamlit Secrets 중 아무거나 사용
- Streamlit Community Cloud/로컬 PC/휴대폰 브라우저 모두 대응
"""

import os
import time
import threading
from datetime import datetime
from typing import List

import streamlit as st
import google.generativeai as genai

# (선택) 로컬 Windows에서만 음성출력(pyttsx3)을 쓸 수 있으므로, 실패해도 앱 계속 동작
try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None


# ==============================
# 0) 공통 설정 및 키 로딩
# ==============================
st.set_page_config(page_title="거울상 챗봇", layout="wide")
st.title("🪞 거울상 챗봇 (Gemini 2.5 Pro)")

API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
API_KEY = (API_KEY or "").strip()

def _mask(s: str) -> str:
    return (s[:6] + "..." + s[-4:]) if s and len(s) > 10 else "(none)"

with st.sidebar:
    st.caption("환경 진단")
    st.write("GOOGLE_API_KEY:", "✅" if API_KEY else "❌", _mask(API_KEY))

if not API_KEY:
    st.error(
        "환경변수/Secrets에 GOOGLE_API_KEY가 설정되지 않았습니다.\n\n"
        "• Streamlit Cloud:  Manage app → Settings → Secrets →  GOOGLE_API_KEY=\"키\"\n"
        "• 로컬 Windows:  CMD에서  setx GOOGLE_API_KEY \"키\"  후 새 터미널에서 실행\n"
        "• 로컬 개발 대안: 프로젝트/.streamlit/secrets.toml 에  GOOGLE_API_KEY=\"키\""
    )
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "models/gemini-2.5-pro"


# ==============================
# 1) 음성 출력 (로컬 PC에서만)
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
        def _run():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

speaker = Speaker()


# ==============================
# 2) 로그 기록
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
# 3) 상태 초기화
# ==============================
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "monologue_running" not in st.session_state:
    st.session_state["monologue_running"] = False


# ==============================
# 4) 옵션 UI
# ==============================
st.sidebar.header("⚙️ 옵션")

max_tokens = st.sidebar.number_input("max_new_tokens", min_value=10, max_value=8192, value=1200, step=50)
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 0.9, 0.05)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.9, 0.01)

st.sidebar.caption("참고: Gemini Python SDK는 현재 top_k, repetition_penalty를 직접 지원하지 않습니다.")

tts_enabled = st.sidebar.checkbox("응답 음성 낭독(로컬 PC 전용)", value=False)
mirror_mode = st.sidebar.checkbox("거울상 모드(대조적 은유 변환)", value=False)

c1, c2 = st.sidebar.columns(2)
if c1.button("혼잣말 시작", use_container_width=True):
    st.session_state["monologue_running"] = True
if c2.button("혼잣말 정지", use_container_width=True):
    st.session_state["monologue_running"] = False


# ==============================
# 5) 거울상 변환
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
# 6) 안전한 응답 텍스트 추출기
# ==============================
def extract_text(resp) -> str:
    """
    google-generativeai 응답 객체에서 사람 읽기용 텍스트를 안전하게 꺼낸다.
    - resp.text 접근 시 ValueError가 나는 경우를 대비해 candidates/parts에서 직접 수집
    - 안전차단(prompt_feedback.block_reason)도 메시지로 알려줌
    """
    # 1) .text 시도
    if hasattr(resp, "text"):
        try:
            t = resp.text
            if t:
                return t.strip()
        except ValueError:
            pass  # 아래 수동 수집으로 진행

    # 2) candidates → content.parts[].text 수집
    texts: List[str] = []
    try:
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for p in parts or []:
                t = getattr(p, "text", None)
                if t:
                    texts.append(t)
    except Exception:
        # 수집 실패는 무시하고 다음 단계로
        pass

    if texts:
        return "\n".join(texts).strip()

    # 3) 차단 사유 표기
    pf = getattr(resp, "prompt_feedback", None)
    block = getattr(pf, "block_reason", None) if pf else None
    if block:
        return f"⚠️ 응답이 안전 정책에 의해 차단되었습니다. (사유: {block})"

    return ""  # 완전 빈 응답


# ==============================
# 7) Gemini 호출
# ==============================
def ask_gemini(user_input: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(
            user_input,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )
        text = extract_text(resp)
        if not text:
            text = "응답을 생성하지 못했습니다. 프롬프트를 조금 바꿔 다시 시도해 주세요."
        return text
    except Exception as e:
        return f"⚠️ Gemini 호출 중 오류가 발생했습니다: {e}"


# ==============================
# 8) 입력창 & 처리
# ==============================
user_input = st.text_input("💬 질문 또는 대화를 입력하세요:")

if user_input:
    answer = ask_gemini(user_input)
    if mirror_mode:
        subject = user_input.strip().split()[0]
        answer = mirror_response(subject, answer)

    st.session_state["messages"].append(("user", user_input))
    st.session_state["messages"].append(("assistant", answer))
    write_log(f"USER: {user_input}\nASSISTANT: {answer}")

    if tts_enabled:
        speaker.speak(answer)


# ==============================
# 9) 혼잣말 모드 (가벼운 1틱)
# ==============================
if st.session_state["monologue_running"]:
    st.info("혼잣말 모드 실행 중... (정지 버튼을 누르면 멈춤)")
    time.sleep(1.0)
    prompt = "은은하고 조용한 혼잣말을 한국어로 1~3문장 해줘. '예수님의 평화와 양의 문' 상징을 가볍게 담아."
    answer = ask_gemini(prompt)
    if mirror_mode:
        answer = mirror_response("혼잣말", answer)

    st.session_state["messages"].append(("assistant", answer))
    write_log(f"ASSISTANT(MONO): {answer}")

    if tts_enabled:
        speaker.speak(answer)


# ==============================
# 10) 대화 출력
# ==============================
for role, msg in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"**👤 사용자:** {msg}")
    else:
        st.markdown(f"**🤖 어시스턴트:** {msg}")
