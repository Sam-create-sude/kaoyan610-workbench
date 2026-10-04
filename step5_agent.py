import os
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)


# ===== 三个工具 =====

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city):
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cur = data["current_condition"][0]
        desc = cur["weatherDesc"][0]["value"]
        return f"{city}当前{desc}，气温 {cur['temp_C']} 摄氏度，体感 {cur['FeelsLikeC']} 摄氏度，湿度 {cur['humidity']}%。"
    except Exception as e:
        return f"天气查询失败（{type(e).__name__}），请稍后再试。"


# ===== 知识库：离线建库 =====

def split_into_chunks(text, max_len=400):
    chunks = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        if len(para) <= max_len:
            chunks.append(para)
            continue
        buf = ""
        for s in para.replace("。", "。\n").split("\n"):
            if len(buf) + len(s) <= max_len:
                buf += s
            else:
                if buf:
                    chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)
    return chunks


def get_embedding(text):
    return embed_client.embeddings.create(
        model="BAAI/bge-m3", input=text
    ).data[0].embedding


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    return dot / (math.sqrt(na) * math.sqrt(nb))


with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("正在建向量库……")
chunks = split_into_chunks(text)
chunk_vectors = [get_embedding(c) for c in chunks]
print(f"知识库共 {len(chunks)} 块，向量库建好了。")


def search_knowledge(question, top_k=3):
    q_vec = get_embedding(question)
    scored = []
    for i in range(len(chunks)):
        scored.append((cosine_similarity(q_vec, chunk_vectors[i]), chunks[i]))
    scored.sort(reverse=True)
    hits = scored[:top_k]
    return "以下是知识库中最相关的资料：\n" + "\n\n".join(c for s, c in hits)


# ===== 工具清单：给模型看的说明书 =====

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户问现在几点、今天几号时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气。当用户问某城市天气、气温、下不下雨时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如：杭州、上海"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在个人知识库中检索资料。知识库内容全部关于 RAG 与检索增强技术：概念定义、切分策略、嵌入与向量检索、关键词与向量的差别、召回率与精确率、Top-K 调整、上下文与令牌成本、工具调用边界、失效排查顺序。当问题涉及这些内容时使用；如果问题与这些主题无关，本工具也检不到任何相关内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要在知识库里检索的问题"}
                },
                "required": ["question"],
            },
        },
    },
]


# ===== 派发表：函数名 → 函数本身 =====

TOOL_MAP = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "search_knowledge": search_knowledge,
}


# ===== Agent 主循环 =====

messages = [
    {"role": "system", "content": "你是一个产品经理助手。可以调用工具查时间、查天气，也可以在知识库里检索资料。回答简洁。\n\n重要规则：凡是涉及知识库的问题，你必须先调用 search_knowledge 工具，并且只依据检索到的资料回答。资料里没有提到的内容，直接回答「知识库里没有相关内容」，绝对不要用你自己的通用知识补上去。"},
]

MAX_ROUNDS = 5

while True:
    user_input = input("\n你：")
    if user_input.strip() in ("exit", "quit", "退出"):
        print("对话结束。")
        break

    messages.append({"role": "user", "content": user_input})

    finished = False
    for round_no in range(MAX_ROUNDS):
        response = chat_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("助手：", msg.content)
            finished = True
            break

        print(f"  [第{round_no + 1}轮] 模型要调用：{[tc.function.name for tc in msg.tool_calls]}")

        for tc in msg.tool_calls:
            fn = TOOL_MAP.get(tc.function.name)
            if fn is None:
                result = "没有这个工具"
            else:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    result = fn(**args)
                except Exception as e:
                    result = f"工具执行出错：{type(e).__name__}"
            print(f"      → 返回：{str(result)[:60]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    if not finished:
        print("助手：（轮数用尽，先停在这里）")
    print(f"  [历史 {len(messages)} 条 | prompt_tokens={response.usage.prompt_tokens}]")
