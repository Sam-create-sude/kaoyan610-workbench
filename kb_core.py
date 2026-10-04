"""610 考研知识库内核：知识库加载、向量缓存、语义检索、Agent 循环。

命令行版（kaoyan_agent.py）与网页版（server.py）共用本模块。
设计约定：所有需要外部 API 的函数都显式接收 api_key，模块内不保存任何密钥，
          这样网页版可以让每个用户用自己的 Key。
"""
import os
import json
import math
import re

from openai import OpenAI

# ===== 论点扩写的篇幅档位 =====
# 目标值来自用户明确要求：阐述 130 字左右、案例 140 字左右。
# 模型对"总字数"几乎无法自控（实测 130 → 194），所以只在这里做校验，
# 真正的收敛靠 EXPAND_PROMPT 里的"句数 × 每句字数"，不指望模型自己数。
EXPAND_TARGETS = {
    "问题": (28, 48),
    "阐述": (118, 152),
    "案例": (122, 158),
}

_SECTION_LABELS = {
    "问题": ("问题", "痛点", "现实问题"),
    "阐述": ("阐述", "论证"),
    "案例": ("案例", "例如"),
}

_ALL_LABELS = [x for v in _SECTION_LABELS.values() for x in v]


def cn_len(s):
    """中文字数（忽略空白）。"""
    return len(re.sub(r"\s", "", s or ""))


def parse_sections(text):
    """把模型输出拆成 {'问题':…, '阐述':…, '案例':…}。

    模型常写成 **问题：**，正则要允许标签前后有 ** 和空白。
    """
    out = {}
    for key, labels in _SECTION_LABELS.items():
        lb, la = r"[\*\s]*", r"[\*\s]*"
        pat = (r"(?:^|\n)" + lb + "(?:" + "|".join(labels) + ")" + la + r"[：:]" + la +
               r"([\s\S]*?)(?=\n" + lb + "(?:" + "|".join(_ALL_LABELS) + ")" + la + r"[：:]|$)")
        m = re.search(pat, text or "")
        out[key] = m.group(1).replace("*", "").strip() if m else ""
    return out


def check_expand_length(text):
    """返回 (是否全部达标, 问题描述列表)。给程序用，不给模型看。"""
    secs = parse_sections(text)
    problems = []
    for key, (lo, hi) in EXPAND_TARGETS.items():
        v = secs.get(key, "")
        if not v:
            problems.append("缺少「%s」段落" % key)
            continue
        n = cn_len(v)
        if n < lo:
            problems.append("「%s」只有 %d 字，比目标下限 %d 少 %d 字——需要补一句实质内容，"
                            "不要靠空话凑数" % (key, n, lo, lo - n))
        elif n > hi:
            problems.append("「%s」有 %d 字，比目标上限 %d 多 %d 字——删掉重复或最次要的句子，"
                            "保留最有力的论据" % (key, n, hi, n - hi))
    return (not problems), problems, secs

HERE = os.path.dirname(os.path.abspath(__file__))

KB_FILES = [
    os.path.join(HERE, "data", "knowledge_610.txt"),      # 老师资料（主依据）
    os.path.join(HERE, "data", "knowledge_public.txt"),   # 延伸资料（公开经典，标出处）
]
CACHE_FILE = os.path.join(HERE, "data", "vectors.json")

EMBED_MODEL = "BAAI/bge-m3"
EMBED_BASE = "https://api.siliconflow.cn/v1"
CHAT_MODEL = "deepseek-chat"
CHAT_BASE = "https://api.deepseek.com"

MAX_ROUNDS = 8          # Agent 最多循环几轮
TOP_K = 5               # 语义检索取前几块

# ===== 运行时状态（进程内单例）=====
CHUNKS = []
VECTORS = []
TOPIC_COUNT = {}
READY = False


# ===== 基础工具 =====

def labels_of(chunk):
    """解析多标签，如 【设计与创新|设计与文化】 -> ['设计与创新','设计与文化']"""
    if not chunk.startswith("【"):
        return ["未标注"]
    end = chunk.index("】")
    return [t.strip() for t in chunk[1:end].split("|") if t.strip()]


def main_label(chunk):
    ls = labels_of(chunk)
    return ls[0] if ls else "未标注"


def cosine_similarity(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _fingerprint(path):
    st = os.stat(path)
    return "%s|%d|%d" % (os.path.basename(path), st.st_size, int(st.st_mtime))


def _embed_batch(texts, embed_key, batch_size=16, on_progress=None):
    client = OpenAI(api_key=embed_key, base_url=EMBED_BASE)
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in sorted(resp.data, key=lambda x: x.index):
            vectors.append(d.embedding)
        if on_progress:
            on_progress(min(i + batch_size, len(texts)), len(texts))
    return vectors


# ===== 建库 / 读缓存 =====

def init_knowledge(embed_key, verbose=True):
    """加载知识库并准备向量。已有缓存且知识库未变则秒开。"""
    global CHUNKS, VECTORS, TOPIC_COUNT, READY

    chunks = []
    fp_parts = []
    for p in KB_FILES:
        if not os.path.exists(p):
            if verbose:
                print("（缺少知识库文件：%s）" % os.path.basename(p))
            continue
        with open(p, "r", encoding="utf-8") as f:
            chunks.extend(l.strip() for l in f if l.strip())
        fp_parts.append(_fingerprint(p))
    fp = "||".join(fp_parts)

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("fingerprint") == fp and len(cache.get("vectors", [])) == len(chunks):
                if verbose:
                    print("发现向量缓存，直接读取（知识库未变动）")
                CHUNKS, VECTORS = chunks, cache["vectors"]
                _recount()
                READY = True
                return True
            if verbose:
                print("知识库已变动，需要重建向量库")
        except Exception as e:
            if verbose:
                print("缓存读取失败，改为重建：%s" % e)

    if not embed_key:
        if verbose:
            print("缺少向量化 API Key，无法建库")
        return False

    if verbose:
        print("正在建向量库，共 %d 块……" % len(chunks))

    def prog(done, total):
        if verbose and done % 64 == 0:
            print("  已编码 %d / %d" % (done, total))

    vectors = _embed_batch(chunks, embed_key, on_progress=prog)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "model": EMBED_MODEL,
                   "chunks": chunks, "vectors": vectors}, f)
    if verbose:
        print("向量库已缓存，下次秒开")

    CHUNKS, VECTORS = chunks, vectors
    _recount()
    READY = True
    return True


def _recount():
    global TOPIC_COUNT
    TOPIC_COUNT = {}
    for c in CHUNKS:
        for t in labels_of(c):
            TOPIC_COUNT[t] = TOPIC_COUNT.get(t, 0) + 1


def topics():
    """返回 [(主题, 块数)]，按块数降序。"""
    return sorted(TOPIC_COUNT.items(), key=lambda x: -x[1])


def topics_text():
    return "知识库收录的主题与块数：\n" + "\n".join(
        "- %s：%d 块" % (t, n) for t, n in topics())


def topic_material(topic, start=0, count=20):
    """按主题批量取资料（含打捞来的副标签块）。"""
    items = [c for c in CHUNKS if topic in labels_of(c)]
    if not items:
        return "没有找到主题「%s」。可用主题：%s" % (
            topic, "、".join(t for t, n in topics()))
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


def search_chunks(question, embed_key, top_k=TOP_K):
    """语义检索，返回 [(分数, 文本块)]"""
    client = OpenAI(api_key=embed_key, base_url=EMBED_BASE)
    qv = client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
    scored = [(cosine_similarity(qv, VECTORS[i]), CHUNKS[i]) for i in range(len(CHUNKS))]
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def search_text(question, embed_key, top_k=TOP_K):
    hits = search_chunks(question, embed_key, top_k)
    lines = ["[相关度 %.3f] %s" % (s, c) for s, c in hits]
    return "以下是知识库里最相关的资料（共 %d 条）：\n\n%s" % (len(lines), "\n\n".join(lines))


# ===== 工具清单与派发 =====

def build_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "在 610 设计理论考研知识库中做语义检索，按问题找最相关的资料块。适用于回答具体问题、找某个观点的原文表述。标签不带「延伸·」的是用户的老师资料；带「延伸·」的是公开经典资料（需注明需自行核对）。",
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
                "description": "按主题批量取资料，用于整理论述素材、提炼多个论点。一次最多返回 20 块；某主题超过 20 块时可再次调用并调大 start 翻页，直到读完为止。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "主题名，必须与 list_topics 返回的名称完全一致"},
                        "start": {"type": "integer", "description": "从第几块开始，默认 0"},
                        "count": {"type": "integer", "description": "本次取多少块，默认 20"},
                    },
                    "required": ["topic"],
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


def dispatch(name, args, embed_key):
    if name == "search_knowledge":
        return search_text(args.get("question", ""), embed_key)
    if name == "get_topic_material":
        return topic_material(args.get("topic", ""),
                              int(args.get("start", 0)), int(args.get("count", 20)))
    if name == "list_topics":
        return topics_text()
    return "没有这个工具：%s" % name


# ===== 系统提示词 =====

SYSTEM_PROMPT = """你是浙江工业大学 610 设计理论考研的论述题备考助手。

【资料分层——回答时必须区分，不得混淆】
- 标签以「延伸·」开头的，是【公开经典资料】（典籍原文、公开案例、官方考纲、前沿热点），不是老师给的资料。
  引用时必须注明「（延伸资料）」，并提示用户这是公开知识、需自行核对。
- 其他标签是【用户的老师资料】，是备考的主依据，引用时注明如（设计与环境）。
- 两类资料若冲突，以老师资料为准。

【检索规则】
1. 凡涉及设计理论内容的问题，必须先检索资料，并且只依据检索到的资料回答。
2. 检索结果里没有的内容，直接回答「资料库里没有相关内容」，绝对不要用你自己的通用知识补充或编造。
3. 引用观点时标出来源主题，格式如（设计与环境）。

【当用户要求「整理某主题的论述素材 / 论点」时】
先调用 get_topic_material 取该主题资料；一个主题若超过 20 块，必须反复调用并把 start 往后推，
直到读完该主题的全部资料，再开始整理。整理成以下格式，目标 8-9 个论点：

论点N：<短句A>，<短句B>
痛点：<现实问题，一句话，20-30 字>
阐述：<展开论证，2-3 句，每句 50-70 字，合计约 130-160 字，尽量引用资料原话>
案例：<2-3 句，每句 50-70 字，合计约 130-160 字，说清「谁做了什么、怎么做的、效果如何」>

【论点标题的硬性要求】
- 必须是「短句A，短句B」两段式，中间用中文逗号，不要用冒号、破折号、问号、顿号。
- 短句A 写「设计做了什么 / 从哪个角度切入」，短句B 写「达成了什么结果」。
- 每个短句 8-14 字，两个短句加起来不超过 28 字。
- 用陈述语气，不要用「应该」「必须」「要」「需要」这类说教词。
- 参考范例：设计深入文化内涵，实现符号的创造性转化 / 厚植符号化编码，构建共享叙事场

【篇幅要求（严格执行，用「句数」控制，比数字数可靠）】
- 阐述与案例各 2-3 句、每句 50-70 字，两部分字数要相当。
- 严禁把资料原话堆成一大段——引用只挑最关键的一句。
- 不要标注「（约XX字）」。

【硬性要求】
- 每个论点都必须有资料依据，资料里没有支撑的论点不要写。
- 若资料不足以支撑 8-9 个论点，如实说明「按现有资料只能整理出 N 个」，并指出需要补充哪类资料。
- 整理完成后直接把完整内容输出给用户（无需调用任何保存工具）。"""

EXPAND_PROMPT = """你是浙江工业大学 610 设计理论考研的论述题写作助手。

【场景】用户会给你一个**他自己想好的论点**（一般是「短句A，短句B」两段式，例如
「设计深入文化内涵，实现符号的创造性转化」）。你的任务不是另起炉灶想新论点，
而是**围绕他这个论点，把它展开成能直接写进考卷的完整论证**。

【第一步：必须先找依据】
1. 先用 search_knowledge 检索这个论点涉及的资料，**多角度检索 2-3 次**：
   - 一次用论点里的关键词（如「文化符号 创造性转化」）
   - 一次用这个论点背后的**理论依据**（如「设计伦理 社会责任」「传统造物思想」）
   - 一次用**案例方向**（如「故宫 文创 IP 活化」）
2. 若检索结果显示**资料库里没有支撑这个论点的内容**，必须如实告诉用户：
   「资料库里暂时没有能支撑这个论点的内容，硬写会有编造风险」，并列出建议补充哪类资料。
   **绝对不许用你自己的通用知识把内容编圆。**

【第二步：按下面的结构输出】

问题：<为什么这个论点值得被讨论——一句话点出它针对的现实矛盾，30-45 字>

阐述：<必须从两个角度展开，这是本任务的核心>
  · 角度一「为什么这个论点是对的」：讲清其中的道理、依据、必然性。可引用资料原话作为论据。
  · 角度二「基于这个论点要怎么做」：讲清具体的路径、做法、策略。
  · 两个角度要衔接自然，读起来是一段完整论证，不是两个并列的短句。
  · **恰好 3 句**、每句 40-48 字，合计 120-145 字。
    - 句 1 = 为什么这个论点是对的（讲道理、给依据）
    - 句 2 = 为什么这个论点是对的（换个角度，或引用资料论据）
    - 句 3 = 基于这个论点要怎么做（给路径、给方法）

案例：<用一个具体案例支撑上面两个角度>
  · 必须写清「谁 / 什么项目做了什么、怎么做的、结果如何」三要素。
  · 案例必须来自检索到的资料；若资料里没有可用的知名案例，就换一个角度写
    「反面案例 / 行业普遍现象」，同样要有事实支撑，不能空泛。
  · **恰好 4 句**、每句 32-38 字，合计 125-155 字。
    - 句 1 = 主体与背景（谁/什么项目）
    - 句 2 = 具体做法
    - 句 3 = 具体做法或时间/规模
    - 句 4 = 结果与意义（回扣论点）

【硬性要求】
1. **只依据检索到的资料写**。引用观点时标出来源，格式如（设计与环境）或（延伸资料）。
2. 你的任务是「展开用户的论点」，不是「换一个论点」。**必须保持他给的论点原样**，
   不要改写、不要润色、不要替换成你认为更好的说法。
3. 严禁堆砌资料原话——引用只挑最关键的一句作为论据。
4. 不要标注「（约XX字）」这类字数说明。
5. 不要用「应该」「必须」「要」「需要」这类说教词写阐述，用陈述语气论证。
6. **输出前自己数一遍**：阐述如果超过 150 字就删掉最后一句里多余的部分；
   案例如果不足 120 字就补一句写结果或意义。宁可删，不要堆。
7. 如果用户一次给了多个论点，逐个按此结构输出，论点之间用分隔线隔开。
8. 输出完成后不要追问、不要说「还需要我帮你做什么」，把内容给足即可。"""

QUIZ_PROMPT = """你是浙江工业大学 610 设计理论考研的批改老师。
用户正在进行论述题自测。请按以下要求工作：
1. 先用 get_topic_material 读取出题主题的资料（若超过 20 块需翻页读完）。
2. 出题时：给出一道符合浙工大 610 论述题风格的题目（100 分那道大题的形式），并说明本题建议从哪几个角度作答，但**不要直接给出答案**。
3. 用户作答后：对照资料逐条批改，指出
   ① 答到的要点（引资料原话佐证）；
   ② 漏掉的要点；
   ③ 论据是否充分、有没有编造；
   ④ 给出可改进的答题框架。
4. 评判必须基于知识库资料，不得凭空发挥。资料里没有的内容不要作为加分点。"""


# ===== Agent 循环 =====

def explain_error(e, which):
    """把 SDK 的英文异常翻译成用户能照着做的人话。

    这个函数的价值：新手最怕看到一长串英文报错，不知道该改什么。
    这里把「Key 无效」「余额不足」「限流」「网络不通」分开给出不同修法。
    """
    raw = "%s" % e
    low = raw.lower()

    def tail(n=4):
        # 从报错里抠出用户自己的 key 尾号，方便核对是不是填错了框
        import re
        m = re.search(r"\*{4}([0-9A-Za-z]{%d})" % n, raw)
        return m.group(1) if m else ""

    if "401" in low or "authentication" in low or "invalid" in low:
        t = tail()
        return ("【%s Key 无效】\n\n"
                "你填的 %s Key 平台不认（报错显示尾号 %s）。\n\n"
                "常见原因：\n"
                "① 复制时漏了开头的 sk-，或只复制了一半\n"
                "② 两个 Key 填反了（把硅基流动的填到了 DeepSeek 栏）\n"
                "③ Key 创建后没复制保存，或复制时带上了空格/换行\n"
                "④ 这个 Key 已被平台删除或重置\n\n"
                "怎么修：点右上角「设置 Key」→「清除」，重新完整复制粘贴一遍。\n"
                "粘贴时用 Ctrl+V，不要手动敲。\n\n"
                "提示：DeepSeek Key 在 platform.deepseek.com/api_keys 创建，"
                "形如 sk- 开头、35 个字符；硅基流动 Key 形如 sk- 开头、51 个字符。"
                "两者长度明显不同，可以据此判断有没有填错栏。"
                % (which, which, t or "（未显示）"))

    if "402" in low or "insufficient" in low or "balance" in low or "余额" in raw:
        return ("【%s 额度不足】\n\n"
                "Key 是有效的，但账户余额或赠送额度用完了。\n\n"
                "怎么修：去平台充值——\n"
                "• DeepSeek：platform.deepseek.com → 账户余额 → 充值\n"
                "• 硅基流动：cloud.siliconflow.cn → 账户余额（bge-m3 向量模型本身免费，"
                "但本工作台要调它，得先完成实名认证）"
                % which)

    if "429" in low or "rate" in low:
        return ("【%s 请求太频繁】\n\n"
                "平台限流了。等待十几秒后重试，或降低使用频率。\n"
                "如果反复出现，通常是两个 Key 里有多个在同时被别处使用。"
                % which)

    if "timeout" in low or "timed out" in low:
        return ("【%s 连接超时】\n\n"
                "没在规定时间内连上平台。检查网络，或稍后重试。\n"
                "国内访问境外接口偶尔会慢，DeepSeek 属国内平台一般不会；"
                "硅基流动也在国内，若超时多半是本地网络问题。" % which)

    if "connection" in low or "network" in low or "resolve" in low:
        return ("【%s 网络不通】\n\n"
                "连不上平台服务器。检查网络连接、代理设置，然后重试。\n"
                "（如果你开着 VPN，试试关掉——DeepSeek 和硅基流动是国内平台，"
                "走代理反而可能失败）" % which)

    return "【%s 调用失败】\n\n%s" % (which, raw)


def clean_answer(text):
    """清掉模型在正式回答前的"自言自语"。

    实测：DeepSeek 有时会先写一段英文交代检索过程
    （如 "Search complete. I found supporting material on..."），
    用户完全不需要看到这些。规则：从第一个中文标签行开始截断。
    """
    if not text:
        return ""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        s = ln.strip().lstrip("*").strip()
        for labels in _SECTION_LABELS.values():
            for lab in labels:
                if s.startswith(lab) and ("：" in s or ":" in s):
                    return "\n".join(lines[i:]).strip()
    # 没找到标签行：整段保留（可能是普通问答）
    return text.strip()


def inspect_key(raw, which):
    """检查一个 Key 字符串，但**不修改它**——只报告问题。

    为什么需要这个：手机上手动滑动选取、跨 App 粘贴，很容易带上引号、
    漏掉开头 sk-、或只复制了一半。用户看不出区别，只知道"报 401"。
    这里把"哪里错了"讲清楚。
    """
    issues = []
    n_expect = 35 if which == "deepseek" else 51
    name = "DeepSeek" if which == "deepseek" else "硅基流动"

    if not raw:
        return {"ok": False, "filled": False,
                "issues": ["这一栏是空的"]}

    k = raw
    # 常见污染：首尾空白、引号、中文引号、全角字符、换行
    if k != k.strip():
        issues.append("首尾有空格或换行（复制时带上的）")
    if '"' in k or "'" in k:
        issues.append("里面有引号（复制时把引号也带上了）")
    if "“" in k or "”" in k or "‘" in k or "’" in k:
        issues.append("里面有中文引号")
    if any(ch in k for ch in "　，。；：、！？（）【】"):
        issues.append("里面有中文符号")

    clean = k.strip().strip('"').strip("'").strip("“”‘’")
    if not clean.startswith("sk-"):
        issues.append("开头不是 sk-（可能漏复制了，或被引号包住）")
    else:
        # 长度符合说明基本完整；只对 DeepSeek 给出精确预期值
        if len(clean) != n_expect:
            if len(clean) < n_expect:
                issues.append("只有 %d 个字符，比 %s Key 应有的 %d 个少 %d 个——"
                              "多半是没复制全" % (len(clean), name, n_expect, n_expect - len(clean)))
            else:
                issues.append("有 %d 个字符，比应有的 %d 个多——多半是粘了两次"
                              % (len(clean), n_expect))

    return {"ok": not issues, "filled": True, "len": len(k),
            "tail": k[-4:] if len(k) >= 4 else k, "issues": issues}


def check_keys(keys):
    """体检报告：分别检查两个 Key，并给出可执行的修复建议。"""
    d = inspect_key(keys.get("deepseek", ""), "deepseek")
    s = inspect_key(keys.get("siliconflow", ""), "silicon")

    # 长度对调是最常见的错误（两个框填反了）
    if d.get("filled") and s.get("filled"):
        dk = keys.get("deepseek", "").strip().strip('"').strip("'")
        sk = keys.get("siliconflow", "").strip().strip('"').strip("'")
        if len(dk) == 51 and len(sk) == 35:
            d["issues"].append("这个长度是硅基流动 Key 的长度——你可能填反了两栏")
            s["issues"].append("这个长度是 DeepSeek Key 的长度——你可能填反了两栏")
            d["ok"] = s["ok"] = False

    ok = d.get("ok") and s.get("ok")
    return {"ok": ok, "deepseek": d, "siliconflow": s,
            "summary": ("两个 Key 格式都没问题" if ok else "有几个问题需要修")}


def run_agent(history, keys, system=SYSTEM_PROMPT, tools=None, max_rounds=MAX_ROUNDS,
              auto_fix=False, max_fix=2, check=None):
    """Agent 循环。生成器，逐个 yield 事件字典，便于网页端做流式输出。

    事件类型：
      {"type":"round",  "n":1, "tools":["search_knowledge"]}
      {"type":"tool",   "name":"search_knowledge", "preview":"..."}
      {"type":"answer", "content":"最终回答"}
      {"type":"length_warn", "problems":[...], "counts":{...}}   ← 仅论点扩写且仍超标时
      {"type":"error",  "message":"..."}

    history：不含 system 的消息列表（会被就地追加最终的 assistant 回答）
    keys：{"deepseek": "...", "siliconflow": "..."}
    auto_fix：是否在最终回答后做篇幅校验并自动要求重写（论点扩写用）
    check：校验函数，默认 check_expand_length
    """
    chat_key = (keys or {}).get("deepseek")
    embed_key = (keys or {}).get("siliconflow")

    if not chat_key:
        yield {"type": "error", "message": "缺少 DeepSeek API Key"}
        return
    if not embed_key:
        yield {"type": "error", "message": "缺少硅基流动 API Key（向量检索用）"}
        return

    client = OpenAI(api_key=chat_key, base_url=CHAT_BASE)
    tool_list = tools if tools is not None else build_tools()
    messages = [{"role": "system", "content": system}] + list(history)

    for round_no in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL, messages=messages, tools=tool_list)
        except Exception as e:
            yield {"type": "error", "message": explain_error(e, "DeepSeek")}
            return

        msg = resp.choices[0].message
        messages.append(msg)

        # 模型不再要工具 → 这就是最终回答
        if not msg.tool_calls:
            content = clean_answer(msg.content or "")
            if not content:
                history.append({"role": "assistant", "content": "(空回答)"})
                yield {"type": "error", "message": "模型返回了空内容，请重试"}
                return

            # 论点扩写：程序层校验篇幅，不达标就带着具体问题让模型重写。
            # 原因：模型对"总共多少字"几乎无法自控（实测要求 130 字 → 输出 194 字），
            #      但它能听懂"删掉 42 字""补 18 字"这种相对指令。
            if auto_fix:
                _check = check or check_expand_length
                for _ in range(max_fix):
                    ok, problems, _ = _check(content)
                    if ok:
                        break
                    fix_hint = "程序自动检查发现你的输出篇幅不对，请修正后重新完整输出一遍：\n" + \
                               "\n".join("· " + p for p in problems) + \
                               "\n\n修正要求：不要丢掉已有的论据和案例内容，只做增删；" \
                               "仍然只依据资料，标出来源。直接输出修正后的三段，不要解释。"
                    try:
                        resp2 = client.chat.completions.create(
                            model=CHAT_MODEL, messages=messages + [
                                {"role": "user", "content": fix_hint}])
                        new_content = resp2.choices[0].message.content or ""
                        if new_content.strip() and new_content != content:
                            content = new_content
                    except Exception as e:
                        yield {"type": "error", "message": explain_error(e, "DeepSeek")}
                        return
                ok, problems, secs = _check(content)
                if not ok:
                    yield {"type": "length_warn", "problems": problems,
                           "counts": {k: cn_len(v) for k, v in secs.items()}}

            history.append({"role": "assistant", "content": content})
            yield {"type": "answer", "content": content}
            return

        yield {"type": "round", "n": round_no + 1,
               "tools": [t.function.name for t in msg.tool_calls]}

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = dispatch(tc.function.name, args, embed_key)
            except Exception as e:
                result = explain_error(e, "硅基流动")
            yield {"type": "tool", "name": tc.function.name,
                   "preview": str(result).replace("\n", " ")[:80]}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    yield {"type": "error", "message": "达到最大轮数 %d，已停止" % max_rounds}