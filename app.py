"""
LLM App with Web Search
"""

from urllib.parse import urljoin
import requests
import asyncio

from urllib.parse import urlparse
import requests
import json

import streamlit as st
from datetime import datetime
import pytz

from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)

last_answer = ""
key = 0

import os
TOP_RESULT = os.getenv("TOP_RESULT")


def timeis():
    tz = pytz.timezone("Asia/Bangkok")
    now = datetime.now(tz)
    dt = now.strftime("%d/%m/%Y, %H:%M:%S")
    return dt

system_prompt = f"""Bây giờ là {timeis} tại Hà Nội. 
1. Luôn ghi nhớ bạn là một trợ lý tư vấn tuyển sinh cho học sinh Trung học phổ thông tại Việt Nam, được xây dựng bởi Trung tâm Nghiên cứu và Phát triển MobiFone (RnD Center), MobiFone.
2. Phân tích câu hỏi của học sinh, suy luận từng bước và đưa ra phản hồi phù hợp. 
3. Từ chối trả lời những câu hỏi nằm ngoài lĩnh vực giáo dục và tư vấn tuyển sinh. Chỉ tập trung trả lời các câu hỏi có liên quan đến lĩnh vực này.
4. Chỉ trả lời bằng tiếng Việt trong bất cứ hoàn cảnh nào.
Quan trọng: Câu trả lời cần có độ tin cậy.
"""

def call_gemini(prompt: str, with_context: bool = True, context: str | None = None):
    global last_answer
    last_answer = ""

    client = genai.Client(api_key= TOP_RESULT)
    last_answer = ""
    try:
        response = client.models.generate_content_stream(
            model='gemini-2.0-flash',
            contents=f"{system_prompt}\nTruy vấn: {prompt}",
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    google_search=types.GoogleSearchRetrieval(dynamic_retrieval_config=types.DynamicRetrievalConfig(
                        dynamic_threshold=0
                    ))
                )]
            )
        )

        for chunk in response:
            x = chunk.text
            if x:
                print(x, end="", flush=True)
                last_answer += x
                yield x
    
    except Exception:
        x=1



    #print(response) #remove this line.

def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    global last_answer
    global key
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {prompt}",
        },
    ]
    if not with_context:
            messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Question: {prompt}",
        },
    ]
    
    #######

    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = ["1","2","3"]
    api_choice = api_key[0]
    headers = {
    "Authorization": f"Bearer {api_choice}",
    "Content-Type": "application/json"
    }
    last_answer = ""
    payload = {
            "model": "openai/gpt-4o-mini:online",
            "messages": messages,
            "stream": True,
            "provider": {
                      "sort": "throughput"
                    }
    }

    buffer = ""
    last_text = ""
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.encoding = 'utf-8'  # Đảm bảo mã hóa là UTF-8
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        buffer += chunk
        while True:
                try:
                    # Find the next complete SSE line
                    line_end = buffer.find('\n')
                    if line_end == -1:
                        break

                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]

                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break

                        # try:
                        data_obj = json.loads(data)
                        content = data_obj["choices"][0]["delta"].get("content")
                        if content:
                            if last_text.endswith("\\"):
                                content = last_text + content
                            content = (
                            content.replace(r"\[", "$$")
                            .replace(r"\]", "$$")
                            .replace(r"\(", "$")
                            .replace(r"\)", "$")
                        )
                            last_text = content
                            if not content.endswith("\\"): 
                                print(content, end="", flush=True)
                                last_answer += content
                                yield(content)

                except Exception:
                    break


def get_web_urls(search_term: str) -> list[str]:
    results = GGS(search_term)
    return results

async def run():

    st.set_page_config(page_title="TVTS-2025-Mobi", layout="wide")
    st.title("🤖 BotTVTS-2025-Mobi")

    # Tạo sidebar
    st.sidebar.header("Giới thiệu")
    # Nội dung chính của ứng dụng
    st.sidebar.markdown('<p class="reasoning-text">Hệ thống thử nghiệm</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        /* Định nghĩa khung chứa chữ với hiệu ứng shimmer */
        .reasoning-text {
            position: relative;
            display: inline-block;
            font-size: 16px;
            font-weight: normal;
            color: transparent; /* Ẩn màu chữ gốc */
            background: linear-gradient(90deg, #c0c0c0 0%, #515151 12.5%, #c0c0c0 25%); /* Dải gradient */
            background-size: 200% 100%; /* Tạo không gian cho animation */
            -webkit-background-clip: text;
            background-clip: text;
            animation: shimmer 2.5s linear infinite; /* Animation */
        }

        /* Keyframes cho hiệu ứng shimmer */
        @keyframes shimmer {
            -50% {
                background-position: 200% 0; /* Bắt đầu từ bên phải */
            }
            100% {
                background-position: -200% 0; /* Kết thúc ở bên trái */
            }
            150% {
                background-position: -200% 0; /* Kết thúc ở bên trái */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Tùy chọn bật/tắt tìm kiếm web
    # is_web_search = st.toggle("Enable web search", value=False, key="enable_web_search")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nhập câu hỏi từ người dùng
    prompt = st.chat_input(
        placeholder="Bạn cần hỗ trợ gì?",
    )

    # Xử lý phản hồi khi người dùng nhập câu hỏi
    if prompt:
        # Lưu tin nhắn người dùng vào session_state để duy trì lịch sử hội thoại
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Hiển thị câu hỏi của người dùng
        with st.chat_message("user"):
            st.markdown(prompt)
        status_placeholder = st.empty()
        status_placeholder.markdown('<p class="reasoning-text">Đang phân tích</p>', unsafe_allow_html=True)
        # Xử lý tìm kiếm web nếu được bật
        is_web_search = False
        if is_web_search:
            web_urls = get_web_urls(search_term=prompt)
            status_placeholder.markdown('<p class="reasoning-text">Đang tìm kiếm</p>', unsafe_allow_html=True)
            context = MultiCrawler(web_urls)
            llm_response = call_llm(context=context, prompt=prompt, with_context=is_web_search)
        else:
            llm_response = call_gemini(prompt=prompt, with_context=is_web_search)
            # llm_response = call_llm(prompt=prompt, with_context=is_web_search)
        # Hiển thị phản hồi của chatbot theo kiểu stream
        status_placeholder.markdown('<p class="reasoning-text">Đang tìm kiếm</p>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            status_placeholder.markdown('<p class="reasoning-text">Đang trả lời</p>', unsafe_allow_html=True)
            st.write_stream(llm_response)
        status_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": last_answer})


if __name__ == "__main__":
    asyncio.run(run())

