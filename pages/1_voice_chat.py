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

from prompts.prompt import VOICE_LLM_PROMPT

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="voice_chat",
    page_icon="🎤"
)
st.title("voice_chat (LLM + OpenAI TTS)")



# Streamlit용 채팅 히스토리 설정
@st.cache_resource
def get_chat_history():
    return StreamlitChatMessageHistory(key="voice_chat_messages")

def get_session_history(session_id: str):
    return get_chat_history()



def parse_llm_response(response: str):
    """
    LLM의 응답에서 '[대답]' 부분과 '[프롬프트]' 부분을 각각 추출
    """
    import re
    # [대답] ... --- [프롬프트] ... 패턴을 파싱
    answer = ""
    voice_prompt = ""
    # 패턴에 맞게 정규식 추출
    match = re.search(
        r"\[대답\]\s*(.*?)\s*-{3,}\s*\[프롬프트\]\s*(.*)", response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        voice_prompt = match.group(2).strip()
    else:
        # 혹시 패턴이 안 맞으면 전체 응답을 대답으로 사용
        answer = response.strip()
        voice_prompt = ""
    return answer, voice_prompt


# OpenAI 챗봇 설정 ------------------------------------------------------
def create_chatbot(model_name="gpt-3.5-turbo", temperature=0.7):
    prompt = ChatPromptTemplate.from_messages([
        ("system", VOICE_LLM_PROMPT),
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
async def generate_tts_wav(text, voice="alloy", instructions=""):
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
    
    # 음성 캐릭터 설정
    st.subheader("🔊 음성 캐릭터 설정")
    voice_style = st.radio(
        "음성 스타일",
        ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"],
        index=0
    )
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화"):
        get_chat_history().clear()
        st.rerun()


    st.markdown("---")
    st.markdown("💡 **계속 같은 감정의 목소리가 아니라 상황별로 감정이 섞인 톤으로 대화합니다.**")
    st.markdown("🔊 **화난 감정일때 어떤 말투냐고 물어보세요.**")

# -------------------------------------

# 채팅 히스토리 가져오기
msgs = get_chat_history()

# 환영 메시지 (최초 접속 시)
if len(msgs.messages) == 0:
    with st.chat_message("assistant"):
        welcome_msg = "또 보네? 또 봐서 기뻐 😊"
        st.markdown(welcome_msg)

# 대화 내용을 화면에 표시 - 히스토리 출력
for message in msgs.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

chatbot = create_chatbot(
    model_name=model_name, 
    temperature=temperature
    )


# ------------------------------------------------------------------------------------------


# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요..."):

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        st.stop()
    
    # 사용자 메시지 표시
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
            
            # 2. 응답 파싱: answer, voice_prompt로 분리
            answer, voice_prompt = parse_llm_response(response)
            
            # 타이핑 효과
            displayed_response = ""
            for word in answer.split():
                displayed_response += word + " "
                time.sleep(0.05)
                message_placeholder.markdown(displayed_response + "▌")
            # 최종 답변
            message_placeholder.markdown(answer)

            # 3. OpenAI TTS로 음성 변환 및 재생
            st.info("AI 답변을 음성으로 듣는 중...")
            if not voice_prompt:
                voice_prompt = "You are a kind AI love partner." # Fallback prompt
            wav_path = asyncio.run(generate_tts_wav(answer, voice=voice_style, instructions=voice_prompt))
            audio_file = open(wav_path, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            audio_file.close()
            os.remove(wav_path)
            
            # base64 인코딩
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                브라우저가 audio 태그를 지원하지 않습니다.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            
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
