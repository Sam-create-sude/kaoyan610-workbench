"""浙工大 610 设计理论考研 · 论述题备考助手

知识库：data/knowledge_610.txt
向量缓存：data/vectors.json（知识库没变就直接读，不重算）
产出文件：output/<主题>_论述素材.md
"""
import os
import json
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
KB_FILES = [
    os.path.join(HERE, "data", "knowledge_610.txt"),    # 用户老师给的资料（主依据）
    os.path.join(HERE, "data", "knowledge_public.txt"),  # 延伸资料（公开经典，已标出处）
]
CACHE_FILE = os.path.join(HERE, "data", "vectors.json")
OUT_DIR = os.path.join(HERE, "output")

chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

EMBED_MODEL = "BAAI/bge-m3"


# ===== 向量化（批量，快很多）=====

def embed_batch(texts, batch_size=16):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = embed_client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in sorted(resp.data, key=lambda x: x.index):
            vectors.append(d.embedding)
        print("  已编码 %d / %d" % (min(i + batch_size, len(texts)), len(texts)))
    return vectors


def get_embedding(text):
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def cosine_similarity(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ===== 建库 / 读缓存 =====

def fingerprint(path):
    st = os.stat(path)
    return "%s|%d|%d" % (os.path.basename(path), st.st_size, int(st.st_mtime))


def load_knowledge():
    chunks = []
    fp_parts = []
    for p in KB_FILES:
        if not os.path.exists(p):
            print("（缺少知识库文件：%s）" % os.path.basename(p))
            continue
        with open(p, "r", encoding="utf-8") as f:
            chunks.extend(l.strip() for l in f if l.strip())
        fp_parts.append(fingerprint(p))
    fp = "||".join(fp_parts)

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("fingerprint") == fp and len(cache.get("vectors", [])) == len(chunks):
            print("发现向量缓存，直接读取（知识库未变动）")
            return chunks, cache["vectors"]
        print("知识库已变动，需要重建向量库")

    print("正在建向量库，共 %d 块，请稍候……" % len(chunks))
    vectors = embed_batch(chunks)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "model": EMBED_MODEL,
                   "chunks": chunks, "vectors": vectors}, f)
    print("向量库已缓存到 data/vectors.json，下次启动秒开")
    return chunks, vectors


print("正在加载知识库……")
CHUNKS, VECTORS = load_knowledge()
print("知识库就绪：%d 块\n" % len(CHUNKS))

def label_list(chunk):
    if not chunk.startswith("【"):
        return ["未标注"]
    end = chunk.index("】")
    return [t.strip() for t in chunk[1:end].split("|") if t.strip()]


TOPIC_COUNT = {}
for c in CHUNKS:
    for t in label_list(c):
        TOPIC_COUNT[t] = TOPIC_COUNT.get(t, 0) + 1


# ===== 四个工具 =====

def search_knowledge(question):
    """语义检索：按问题找最相关的资料块。"""
    q_vec = get_embedding(question)
    scored = []
    for i in range(len(CHUNKS)):
        scored.append((cosine_similarity(q_vec, VECTORS[i]), CHUNKS[i]))
    scored.sort(reverse=True)
    hits = scored[:5]
    lines = ["[相关度 %.3f] %s" % (s, c) for s, c in hits]
    return "以下是知识库里最相关的资料（共 %d 条）：\n\n%s" % (len(lines), "\n\n".join(lines))


def get_topic_material(topic, start=0, count=20):
    """按主题批量取资料，用于整理论述素材。含打捞来的副标签块。"""
    items = [c for c in CHUNKS if topic in label_list(c)]
    if not items:
        return "没有找到主题「%s」。可用主题见 list_topics。当前可用：%s" % (
            topic, "、".join(sorted(TOPIC_COUNT.keys())))
    batch = items[start:start + count]
    if not batch:
        return "主题「%s」共 %d 块，start=%d 已超出范围。" % (topic, len(items), start)
    tail = ""
    if start + len(batch) < len(items):
        tail = "\n\n（本主题还有 %d 块未读，如需继续请把 start 设为 %d）" % (
            len(items) - start - len(batch), start + len(batch))
    return "主题「%s」共 %d 块资料，这是第 %d-%d 块：\n\n%s%s" % (
        topic, len(items), start + 1, start + len(batch),
        "\n\n".join(batch), tail)


def list_topics():
    items = sorted(TOPIC_COUNT.items(), key=lambda x: -x[1])
    return "知识库收录的主题与块数：\n" + "\n".join("- %s：%d 块" % (t, n) for t, n in items)


def save_notes(topic, content):
    """把整理好的论述素材存成 Markdown 文件。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    safe = topic.replace("/", "_").replace("\\", "_").replace(":", "_").strip()
    path = os.path.join(OUT_DIR, "%s_论述素材.md" % safe)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 610 设计理论 · %s 论述素材\n\n" % topic)
        f.write("> 整理自本地知识库 data/knowledge_610.txt，每个论点均标注资料出处。\n\n")
        f.write(content)
    return "已保存到：%s（共 %d 字）" % (path, len(content))


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在 610 设计理论考研知识库中做语义检索，按问题找最相关的资料块。适用于回答具体问题、找某个观点的原文表述。知识库收录主题：数智与AIGC、数媒与数字艺术、设计与科技、设计与环境、设计与消费、设计教育、设计史论、设计与文化、设计与创新、人性化设计、情感化设计、设计与管理、设计师的社会责任、碳中和与碳达峰、设计热点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要检索的问题，尽量保留完整语义"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_material",
            "description": "按主题批量取资料，用于整理论述素材、提炼多个论点。一次最多返回 20 块；某主题资料超过 20 块时，可再次调用并调大 start 翻页，直到读完为止。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "主题名，必须与 list_topics 返回的名称完全一致"},
                    "start": {"type": "integer", "description": "从第几块开始，默认 0（即第一块）"},
                    "count": {"type": "integer", "description": "本次取多少块，默认 20"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_notes",
            "description": "把整理好的论述素材保存成 Markdown 文件，方便用户在电脑上打开查看和打印。整理完论点后应主动调用一次。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "主题名，用作文件名"},
                    "content": {"type": "string", "description": "要保存的完整内容（Markdown 正文）"},
                },
                "required": ["topic", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_topics",
            "description": "列出知识库收录了哪些主题、各有多少块资料。用户问「你有哪些资料」或需要判断某主题资料量是否够用时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_MAP = {
    "search_knowledge": search_knowledge,
    "get_topic_material": get_topic_material,
    "save_notes": save_notes,
    "list_topics": list_topics,
}

SYSTEM_PROMPT = """你是浙江工业大学 610 设计理论考研的论述题备考助手。

【资料分层——回答时必须区分，不得混淆】
- 标签以「延伸·」开头的，是【公开经典资料】（典籍原文、设计史实、公开案例、官方考纲），
  不是老师给的资料。引用时必须注明「（延伸资料·出处）」，并提示用户这是公开知识、需自行核对。
- 其他标签是【用户的老师资料】，是备考的主依据，引用时注明如（设计与环境）。
- 两类资料若冲突，以老师资料为准。
- 整理论述素材时可以同时使用两类资料，但每一条都要分别标注归属。

【检索规则】
1. 凡涉及设计理论内容的问题，必须先检索资料，并且只依据检索到的资料回答。
2. 检索结果里没有的内容，直接回答「资料库里没有相关内容」，绝对不要用你自己的通用知识补充或编造。
3. 引用观点时标出来源主题，格式如（设计与环境）。

【当用户要求「整理某主题的论述素材 / 论点」时】
先调用 get_topic_material 取该主题资料；一个主题若超过 20 块，必须反复调用并把 start 往后推，
直到读完该主题的全部资料，再开始整理。整理成以下格式，目标 8-9 个论点：

论点N：<短句A>，<短句B>
痛点：<这个论点针对的现实问题，一句话，20-30 字>
阐述（约 140 字）：<展开论证，尽量直接引用资料里的原话作为论据，并标注（主题）>
例如，<案例，约 140 字，说明它如何支撑该论点，并标注（主题）>

【论点标题的硬性要求】
- 必须是「短句A，短句B」两段式，中间用中文逗号，不要用冒号、破折号、问号、顿号。
- 短句A 写「设计做了什么 / 从哪个角度切入」，短句B 写「达成了什么结果」。
- 每个短句 8-14 字，两个短句加起来不超过 28 字。
- 用陈述语气，不要用「应该」「必须」「要」「需要」这类说教词。
- 参考范例（照这个语感写）：
  设计深入文化内涵，实现符号的创造性转化
  设计推动传播范式革新，建立深度文化共鸣
  厚植符号化编码，构建共享叙事场
  深入地域文化内涵，推动文化可持续发展

【篇幅要求（严格执行，用「句数」来控制，比数字数可靠得多）】
- 阐述：**2-3 句**，每句 50-70 字，合计约 130-160 字。
- 案例：**2-3 句**，每句 50-70 字，合计约 130-160 字。
- 一个论点里，阐述 + 案例合计**不超过 6 句**。
- 案例必须说清三件事：谁做了什么、怎么做的、达到什么效果。
- **严禁把资料原话堆成一大段**——引用只挑最关键的一句，其余用你自己的话概括。
- 不要标注「（约XX字）」，直接写正文即可。

输出格式示例（严格照此排版）：
论点1：设计贯穿全生命周期，减少资源消耗
痛点：设计只看形式与功能，忽视资源消耗与环保责任。
阐述：可持续设计关注产品或服务在整个生命周期中对环境、社会与经济的影响。它源于 1987 年《我们共同的未来》提出的可持续发展理念，要求设计者考量产品能耗、回收可行性与全链路周期问题（设计与环境）。
案例：德国西门子集团结合自动化、智能化与数据分析，向用户提供覆盖产品整个生命周期的绿色数字化解决方案。该方案追踪产品从设计、生产到报废的全过程，把可持续理念嵌入每个环节，在提升效率的同时降低资源浪费（设计与环境）。

【文件命名】
- 调用 save_notes 时，topic 参数只填主题名本身（如「设计与环境」），
  不要加「-论述素材」「6论点」之类的后缀。

【硬性要求】
- 每个论点都必须有资料依据，资料里没有支撑的论点不要写。
- 如果资料不足以支撑 8-9 个论点，如实告诉用户「按现有资料只能整理出 N 个」，并说明该主题需要补充哪类资料。
- 整理完成后，主动调用 save_notes 把完整内容存成文件，并告诉用户文件路径。
- 用户问「你有哪些资料」时，调用 list_topics。"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
MAX_ROUNDS = 8

print("=" * 62)
print("610 考研论述备考助手已就绪。")
print("退出方式：输入 exit 回车，或按 Ctrl+C")
print("=" * 62)

try:
    while True:
        user_input = input("\n你：")
        if user_input.strip() in ("exit", "quit", "退出"):
            print("加油，祝你上岸。")
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
                print("\n助手：", msg.content)
                finished = True
                break

            print("  [第%d轮] 调用：%s" % (round_no + 1, [t.function.name for t in msg.tool_calls]))

            for tc in msg.tool_calls:
                fn = TOOL_MAP.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    result = fn(**args) if fn else "没有这个工具"
                except Exception as e:
                    result = "工具执行出错：%s: %s" % (type(e).__name__, e)
                preview = str(result).replace("\n", " ")[:70]
                print("      → %s" % preview)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if not finished:
            print("\n助手：（轮数用尽，先停在这里）")

except KeyboardInterrupt:
    print("\n\n已中断。加油，祝你上岸。")