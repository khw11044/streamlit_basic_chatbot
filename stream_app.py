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

from prompts.prompt import SYSTEM_PROMPT

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 웹사이트 제목
st.title("OpenAI Chatbot with Memory")

# Streamlit용 채팅 히스토리 설정
@st.cache_resource
def get_chat_history():
    return StreamlitChatMessageHistory(key="chat_messages")

def get_session_history(session_id: str):
    return get_chat_history()

# OpenAI 챗봇 설정
def create_chatbot(model_name="gpt-3.5-turbo", temperature=0.7):
    # 프롬프트 템플릿 설정
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화"):
        # Streamlit 채팅 히스토리 초기화
        get_chat_history().clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("💡 **동물의 숲 '너굴'과 대화해보세요.")

# 채팅 히스토리 가져오기
msgs = get_chat_history()

# 환영 메시지 (최초 접속 시)
if len(msgs.messages) == 0:
    with st.chat_message("assistant"):
        welcome_msg = "안녕하세요! 저는 대화 내용을 기억할 수 있는 AI 어시스턴트입니다. 무엇을 도와드릴까요? 😊"
        st.markdown(welcome_msg)
        msgs.add_ai_message(welcome_msg)

# 대화 내용을 화면에 표시 (StreamlitChatMessageHistory에서)
for message in msgs.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# 챗봇 설정 (대화 시작 전에 한 번만 생성)
chatbot = create_chatbot(model_name=model_name, temperature=temperature)

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
            🧠 이 챗봇은 대화 내용을 기억합니다. OpenAI API를 사용합니다.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
