import os
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()


def split_into_chunks(text, max_len=500):
    text = text.replace("\n", "")
    text = text.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n")

    sentences = []
    for s in text.split("\n"):
        s = s.strip()
        if s:
            sentences.append(s)

    chunks = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) <= max_len:
            buf = buf + s
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def get_embedding(text):
    response = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        dot = dot + a[i] * b[i]
        norm_a = norm_a + a[i] * a[i]
        norm_b = norm_b + b[i] * b[i]
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def search(query, chunks, chunk_vectors, top_k=3):
    q_vec = get_embedding(query)
    scored = []
    for i in range(len(chunks)):
        score = cosine_similarity(q_vec, chunk_vectors[i])
        scored.append((score, chunks[i]))
    scored.sort(reverse=True)
    return scored[:top_k]


chunks = split_into_chunks(text)
print(f"文档切成了 {len(chunks)} 块")

print("正在把每块编码成向量……")
chunk_vectors = []
for chunk in chunks:
    chunk_vectors.append(get_embedding(chunk))

print(f"编码完成：{len(chunk_vectors)} 块，每块是 {len(chunk_vectors[0])} 个数字的向量")

question = "讲以下离线向量与在线向量的区别？"
hits = search(question, chunks, chunk_vectors)

print(f"\n检索到 {len(hits)} 块相关资料：")
for score, chunk in hits:
    print(f"  相似度 {score:.4f} | {chunk[:40]}...")

context = "\n\n".join(chunk for score, chunk in hits)

prompt = f"""请严格根据下面的资料回答问题。如果资料里没有相关内容，就直接回答"资料里没有提到"。

资料：
{context}

问题：{question}"""

response = chat_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个严谨的资料助手，只依据给定资料回答。"},
        {"role": "user", "content": prompt},
    ],
)
print("\n最终回答：")
print(response.choices[0].message.content)