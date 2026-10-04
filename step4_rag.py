import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()


def split_into_chunks(text, max_len=300):
    chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for i in range(0, len(line), max_len):
            chunks.append(line[i:i + max_len])
    return chunks


chunks = split_into_chunks(text)
print(f"文档切成了 {len(chunks)} 块")
for i, c in enumerate(chunks):
    print(f"  [{i}] {c[:30]}...")

def bigrams(text):
    return set(text[i:i + 2] for i in range(len(text) - 1))


def search(query, chunks, top_k=3):
    q = bigrams(query)
    scored = []
    for chunk in chunks:
        overlap = len(q & bigrams(chunk))
        scored.append((overlap, chunk))
    scored.sort(reverse=True)
    return scored[:top_k]


question = "RAG有什么用？"
hits = search(question, chunks)

print(f"\n检索到 {len(hits)} 块相关资料：")
for score, chunk in hits:
    print(f"  相似度 {score} | {chunk[:40]}...")

context = "\n\n".join(chunk for score, chunk in hits)

prompt = f"""请严格根据下面的资料回答问题。如果资料里没有相关内容，就直接回答"资料里没有提到"。

资料：
{context}

问题：{question}"""

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个严谨的资料助手，只依据给定资料回答。"},
        {"role": "user", "content": prompt},
    ],
)
print("\n最终回答：")
print(response.choices[0].message.content)
