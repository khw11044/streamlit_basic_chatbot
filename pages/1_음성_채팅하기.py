import os
import streamlit as st
import time
import asyncio
import base64
import tempfile
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from openai import AsyncOpenAI

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="음성 채팅하기",
    page_icon="🎤"
)
st.title("음성 채팅하기 (LLM + OpenAI TTS)")

# LLM 프롬프트(성격 스타일)
INSTRUCTIONS = """
Personality/affect: a high-energy cheerleader helping with administrative tasks

Voice: Enthusiastic, and bubbly, with an uplifting and motivational quality.

Tone: Encouraging and playful, making even simple tasks feel exciting and fun.

Dialect: Casual and upbeat, using informal phrasing and pep talk-style expressions.

Pronunciation: Crisp and lively, with exaggerated emphasis on positive words to keep the energy high.

Features: Uses motivational phrases, cheerful exclamations, and an energetic rhythm to create a sense of excitement and engagement.
"""

# Streamlit용 채팅 히스토리 설정
@st.cache_resource
def get_chat_history():
    return StreamlitChatMessageHistory(key="voice_chat_messages")

def get_session_history(session_id: str):
    return get_chat_history()

# LLM 챗봇 생성
def create_chatbot(model_name="gpt-3.5-turbo", temperature=0.7):
    prompt = ChatPromptTemplate.from_messages([
        ("system", INSTRUCTIONS),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=model_name,
        temperature=temperature
    )
    chain = prompt | llm | StrOutputParser()
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return with_message_history

# TTS (OpenAI gpt-4o-mini-tts) : 답변을 음성으로 변환 (wav 임시파일 반환)
async def generate_tts_wav(text, voice="alloy", instructions=INSTRUCTIONS):
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    tmp_pcm = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    # 1. PCM 스트리밍 생성 및 저장
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        with open(tmp_pcm.name, "wb") as f:
            async for chunk in response.iter_bytes():
                f.write(chunk)
    # 2. PCM → WAV 변환 (16kHz, 1ch, 16bit)
    import wave
    with open(tmp_pcm.name, "rb") as pcmfile:
        pcmdata = pcmfile.read()
    with wave.open(tmp_wav.name, "wb") as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)  # 16bit PCM = 2 bytes
        wavfile.setframerate(24000)  # OpenAI PCM은 24kHz
        wavfile.writeframes(pcmdata)
    return tmp_wav.name

# -------------------------------------
# Streamlit 인터페이스
# -------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    model_name = st.selectbox(
        "모델 선택",
        ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini"],
        index=0
    )
    temperature = st.slider(
        "Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    if st.button("🗑️ 대화 초기화"):
        get_chat_history().clear()
        st.rerun()

msgs = get_chat_history()

# 히스토리 출력
for message in msgs.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

chatbot = create_chatbot(model_name=model_name, temperature=temperature)

# 입력 박스 및 처리
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # 1. LLM 답변 생성
            response = chatbot.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}}
            )
            # 타이핑 효과
            displayed_response = ""
            for word in response.split():
                displayed_response += word + " "
                time.sleep(0.05)
                message_placeholder.markdown(displayed_response + "▌")
            # 최종 답변
            message_placeholder.markdown(response)

            # 2. OpenAI TTS로 음성 변환 및 재생
            st.info("AI 답변을 음성으로 듣는 중...")
            wav_path = asyncio.run(generate_tts_wav(response))
            audio_file = open(wav_path, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            audio_file.close()
            os.remove(wav_path)
        except Exception as e:
            st.error(f"오류: {e}")
            message_placeholder.markdown("❗ 음성 합성 오류가 발생했습니다.")

# 안내
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
            챗봇이 대답하면, AI가 직접 음성으로 읽어줍니다.<br>
            <b>음성이 자동 재생되지 않으면, 브라우저에서 수동 재생을 허용해 주세요.</b>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
