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

google_search_tool = Tool(
    google_search = GoogleSearch()
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
2. Cần đọc hiểu, phân tích ý định của câu hỏi người dùng trước khi tìm kiếm và trả lời nội dung phù hợp. Ví dụ, "điểm chuẩn 3 năm gần nhất" thì phân tích ý định "3 năm gần nhất là 2024, 2023 và 2022 vì hôm nay là {timeis}".
3. Chỉ trả lời bằng tiếng Việt trong bất cứ hoàn cảnh nào.
4. Bỏ qua phần giới thiệu bản thân và các thao tác. Trực tiếp trả lời câu hỏi. Chỉ đưa ra câu trả lời, không nêu ra các bước thực hiện (ví dụ: tôi cần tìm kiếm, ...)
5. Nếu không có hoặc không đủ thông tin có thể trả lời, trực tiếp đưa ra kết quả là không trả lời được câu hỏi này.
Quan trọng: Câu trả lời cần có độ tin cậy và cập nhật mới nhất.
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
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )
        for chunk in response:
            x = chunk.text
            if x:
                print(x, end="", flush=True)
                last_answer += x
                yield x
            # Tích hợp code lấy title và uri từ grounding chunks vào đây
                if chunk and chunk.candidates:
                    candidate = chunk.candidates[0] # Lấy candidate đầu tiên, giả sử có 1 candidate
                    if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks:
                        grounding_chunks = candidate.grounding_metadata.grounding_chunks
                        print("\n---Grounding Chunks Info---") # Thêm dấu hiệu để phân biệt output text và grounding info
                        for grounding_chunk in grounding_chunks:
                            if grounding_chunk.web:
                                web_info = grounding_chunk.web
                                title = web_info.get('title')
                                uri = web_info.get('uri')
                                if title and uri: # Kiểm tra để tránh in ra None nếu không có title hoặc uri
                                    print(f"\n  Title: {title}")
                                    print(f"  URI: {uri}")
                        print("---End Grounding Chunks Info---")
                    else:
                        print("\n---Grounding Chunks Info---")
                        print("  No grounding_chunks found in grounding_metadata for this chunk.")
                        print("---End Grounding Chunks Info---")
                else:
                    print("\n---Grounding Chunks Info---")
                    print("  No candidates found in this chunk.")
                    print("---End Grounding Chunks Info---")

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
            background: linear-gradient(90deg, #c0c0c0 0%, #111111 12.5%, #c0c0c0 25%); /* Dải gradient */
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

