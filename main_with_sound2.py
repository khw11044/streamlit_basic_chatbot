import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# TTS 관련 imports
from gtts import gTTS
from pydub import AudioSegment
import random
import base64
import io

# from prompts.prompt import SYSTEM_PROMPT
from prompts.prompt import PROMPT_DICT

# r2-d2 스타일 임포트
from r2d2 import generate_r2d2_voice

# 웹사이트 제목
st.title("음성 챗봇")

# 디렉토리 생성
@st.cache_data
def create_directories():
    os.makedirs('samples', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    return True

create_directories()

# 너굴이 음성 생성 함수
@st.cache_data
def generate_nook_voice(text, lang='ko', random_factor=0.35, normal_frame_rate=44100):
    """너굴이 스타일 음성 생성 (특수문자/숫자는 짧은 무음으로 처리)"""
    if not text.strip():
        return None

    result_sound = None
    short_silence = AudioSegment.silent(duration=150)  # 0.15초 짧은 무음

    for i, letter in enumerate(text):
        if letter == ' ':  # 공백
            new_sound = AudioSegment.silent(duration=200)  # 0.2초 무음
        elif not (letter.isalpha() or '\uAC00' <= letter <= '\uD7A3'):  # 한글 또는 영문이 아니면 = 특수문자/숫자
            new_sound = short_silence
        else:
            letter_file = f'samples/{letter}.mp3'
            if not os.path.isfile(letter_file):
                try:
                    tts = gTTS(letter, lang=lang)
                    tts.save(letter_file)
                except Exception as e:
                    st.error(f"TTS 생성 실패: {letter} - {e}")
                    continue
            try:
                letter_sound = AudioSegment.from_mp3(letter_file)
                if len(letter_sound.raw_data) > 10000:
                    raw = letter_sound.raw_data[5000:-5000]
                else:
                    raw = letter_sound.raw_data
                    
                octaves = 2.0 + random.random() * random_factor
                frame_rate = int(letter_sound.frame_rate * (2.0 ** octaves))
                new_sound = letter_sound._spawn(raw, overrides={'frame_rate': frame_rate})
                
            except Exception as e:
                st.error(f"음성 처리 실패: {letter} - {e}")
                continue

        if 'new_sound' in locals():
            new_sound = new_sound.set_frame_rate(normal_frame_rate)
            result_sound = new_sound if result_sound is None else result_sound + new_sound

    return result_sound

# 오디오를 base64로 인코딩하여 HTML에서 재생할 수 있게 함
def audio_to_base64(audio_segment):
    """AudioSegment를 base64 문자열로 변환"""
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    audio_base64 = base64.b64encode(buffer.getvalue()).decode()
    return audio_base64

# Streamlit용 채팅 히스토리 설정
@st.cache_resource
def get_chat_history():
    return StreamlitChatMessageHistory(key="chat_messages")

def get_session_history(session_id: str):
    return get_chat_history()

# OpenAI 챗봇 설정
def create_chatbot(model_name="gpt-3.5-turbo", temperature=0.7, voice_style="일반"):
    # 프롬프트 템플릿 설정
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_DICT.get(voice_style, PROMPT_DICT["일반"])),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # ChatOpenAI 모델 설정
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
        temperature=temperature
    )
    
    # 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # 메시지 히스토리와 함께 실행 가능한 체인 생성
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return with_message_history

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모델 선택
    model_name = st.selectbox(
        "모델 선택",
        ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini"],
        index=0
    )
    
    # Temperature 설정
    temperature = st.slider(
        "Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    # 음성 설정
    st.subheader("🔊 음성 설정")
    voice_style = st.radio(
        "음성 스타일",
        ["일반", "너굴", "r2-d2"],
        index=0
    )
    enable_voice = st.checkbox("음성 재생", value=True)
    voice_random_factor = st.slider(
        "음성 변조 강도",
        min_value=0.1,
        max_value=0.8,
        value=0.35,
        step=0.05,
        help="값이 클수록 더 다양한 톤으로 말합니다"
    )
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화"):
        get_chat_history().clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("💡 **동물의 숲 '너굴'과 스타워즈의 'r2-d2'와 대화해보세요. **")
    st.markdown("🔊 **그리고 직접 들어보세요!**")

# 채팅 히스토리 가져오기
msgs = get_chat_history()

# 환영 메시지 (최초 접속 시)
if len(msgs.messages) == 0:
    with st.chat_message("assistant"):
        welcome_msg = "안녕하세요! 저는 챗봇입니다! 무엇을 도와드릴까요? 😊"
        st.markdown(welcome_msg)

# 대화 내용을 화면에 표시
for message in msgs.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# 챗봇 설정
chatbot = create_chatbot(
    model_name=model_name, 
    temperature=temperature,
    voice_style=voice_style
    )

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        st.stop()
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성 및 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # AI 응답 생성
            response = chatbot.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "default"}}
            )
            
            # 타이핑 효과로 응답 표시
            displayed_response = ""
            for word in response.split():
                displayed_response += word + " "
                time.sleep(0.05)
                message_placeholder.markdown(displayed_response + "▌")
            
            # 최종 응답 표시
            message_placeholder.markdown(response)
            
            # 음성 생성 및 재생
            if enable_voice and response.strip():
                with st.spinner(f"{voice_style} 목소리 생성 중..."):
                    try:
                        if voice_style == "일반":
                            tts = gTTS(response, lang='ko')
                            tts_fp = io.BytesIO()
                            tts.write_to_fp(tts_fp)
                            tts_fp.seek(0)
                            audio_seg = AudioSegment.from_file(tts_fp, format="mp3")
                        elif voice_style == "너굴":
                            audio_seg = generate_nook_voice(
                                response, 
                                random_factor=voice_random_factor
                            )
                        else:  # r2-d2
                            base_dir = os.path.dirname(__file__)
                            audio_seg = generate_r2d2_voice(
                                response,
                                base_dir
                            )
                        if audio_seg:
                            audio_base64 = audio_to_base64(audio_seg)
                            audio_html = f"""
                            <audio autoplay>
                                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                            </audio>
                            """
                            st.markdown(audio_html, unsafe_allow_html=True)
                            st.download_button(
                                label="🔊 음성 다운로드",
                                data=base64.b64decode(audio_base64),
                                file_name=f"{voice_style}_voice_{int(time.time())}.mp3",
                                mime="audio/mp3"
                            )
                        else:
                            st.warning("음성 생성에 실패했습니다.")

                    except Exception as e:
                        st.error(f"음성 생성 오류: {str(e)}")
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            error_response = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
            message_placeholder.markdown(error_response)

# 하단 정보
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
            챗봇과 다양한 목소리로 대화해보세요!<br>
            💡 음성이 자동 재생되지 않으면 브라우저 설정에서 자동재생을 허용해주세요.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
