"""知识库增强：修正主题标签 + 跨主题打捞。

两个动作，全部基于用户自己的资料，不生成任何新内容：
  1. 【修正标签】把标成「综合」的块，按内容语义归到最匹配的主题（文件名不可靠）
  2. 【跨主题打捞】资料量不足 15 块的主题，把别的主题里语义高度相关的块
     追加一个副标签（形如 【设计与创新|设计与文化】），让它也能被该主题检索到

标签格式：主主题|副主题1|副主题2
"""
import os
import json
import math
from dotenv import load_dotenv
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

KB = os.path.join(HERE, "data", "knowledge_610.txt")

client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"), base_url="https://api.siliconflow.cn/v1")
MODEL = "BAAI/bge-m3"

RELABEL_TH = 0.60      # 「综合」重新归类的最低相似度
SALVAGE_TH = 0.65      # 跨主题打捞的最低相似度（宁缺毋滥）
THIN_LIMIT = 15        # 少于这么多块的主题算「薄」，需要打捞
MAX_SECONDARY = 2      # 一个块最多挂几个副主题
MAX_SALVAGE_PER_TOPIC = 20   # 每个薄主题最多打捞多少块，防稀释

TOPIC_QUERIES = {
    "数智与AIGC": "人工智能 AIGC 生成式人工智能 智能设计 大数据 机器学习 数智化 智能时代设计师",
    "数媒与数字艺术": "数字媒体艺术 数字技术 交互设计 虚拟现实 新媒体艺术 视觉传达 数字展陈",
    "设计与科技": "设计与科技的关系 科技创新 新技术 新材料 艺术与科技融合 人工智能与设计 智能产品",
    "设计与环境": "可持续设计 绿色设计 生态设计 碳中和 碳达峰 环境保护 全生命周期 低碳 源头干预 末端治理",
    "设计与消费": "设计消费 品牌设计 用户体验 消费需求 商业设计 市场 消费升级 情感消费",
    "设计教育": "设计教育 教学改革 课程体系 设计人才培养 学科建设 艺术设计专业",
    "设计史论": "设计史 设计理论 设计思想 设计学概论 工艺美术史 现代设计运动 设计流派 设计师人物",
    "设计与文化": "设计与文化 传统文化 文化符号 文化转译 非遗活化 本土化设计 文化自信 地方文化 造物思想",
    "设计与创新": "设计创新 创新思维 原创设计 设计驱动创新 创造力 创新策略",
    "人性化设计": "人性化设计 以人为本 人机工程 用户体验 无障碍设计 通用设计 人的需求",
    "情感化设计": "情感化设计 情感体验 本能层 行为层 反思层 用户情感 情感需求 情绪",
    "设计与管理": "设计管理 设计战略 设计流程管理 设计价值 设计组织 品牌管理 设计决策",
    "设计师的社会责任": "设计师的责任 设计伦理 职业道德 社会责任 可持续责任 设计价值观",
    "碳中和与碳达峰": "碳中和 碳达峰 双碳目标 碳排放 低碳设计 节能减排 绿色低碳 碳足迹",
    "设计热点": "当下设计热点 元宇宙 巴黎奥运会 设计趋势 热门话题 最新动态",
    "写作模板": "论述题写作 万能开头 论文结构 答题模板 论述框架",
}


def label_of(chunk):
    if not chunk.startswith("【"):
        return []
    end = chunk.index("】")
    return [t.strip() for t in chunk[1:end].split("|") if t.strip()]


def main_label(chunk):
    ls = label_of(chunk)
    return ls[0] if ls else "未标注"


def rebuild(chunk, labels):
    end = chunk.index("】")
    return "【%s】%s" % ("|".join(labels), chunk[end + 1:])


def embed(texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        r = client.embeddings.create(model=MODEL, input=texts[i:i + batch])
        out.extend(d.embedding for d in sorted(r.data, key=lambda x: x.index))
    return out


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


chunks = [l.strip() for l in open(KB, encoding="utf-8") if l.strip()]
print("读入 %d 块" % len(chunks))

topics = list(TOPIC_QUERIES.keys())
print("正在编码主题概念与全部资料块……")
topic_vecs = embed([TOPIC_QUERIES[t] for t in topics])
chunk_vecs = embed(chunks)

counts = {}
for c in chunks:
    lab = main_label(c)
    counts[lab] = counts.get(lab, 0) + 1
thin = [t for t in topics if counts.get(t, 0) < THIN_LIMIT]
print("资料量不足 %d 块的主题（需要打捞）：%s" % (THIN_LIMIT, "、".join(thin)))
print()

# 第一遍：算相似度 + 修正「综合」标签
sims_all = []
chunk_labels = []
relabeled = 0
for i, c in enumerate(chunks):
    labels = label_of(c)
    sims = [(cos(chunk_vecs[i], topic_vecs[j]), topics[j]) for j in range(len(topics))]
    sims.sort(key=lambda x: -x[0])

    if labels and labels[0] == "综合" and sims[0][0] >= RELABEL_TH:
        labels = [sims[0][1]]
        relabeled += 1
    if not labels:
        labels = [sims[0][1]] if sims[0][0] >= RELABEL_TH else ["综合"]

    sims_all.append(sims)
    chunk_labels.append(labels)

# 第二遍：收集打捞候选，每个薄主题按分数取前 N 块（防止稀释主题）
cands = {t: [] for t in thin}
for i, sims in enumerate(sims_all):
    for score, topic in sims:
        if topic in chunk_labels[i] or topic not in thin:
            continue
        if score >= SALVAGE_TH:
            cands[topic].append((score, i))

salvage_map = {}
for topic in thin:
    cands[topic].sort(key=lambda x: -x[0])
    picked = cands[topic][:MAX_SALVAGE_PER_TOPIC]
    print("  打捞【%s】：候选 %d 块 → 取前 %d 块（最低分 %.3f）" % (
        topic, len(cands[topic]), len(picked), picked[-1][0] if picked else 0))
    for score, i in picked:
        salvage_map.setdefault(i, []).append(topic)

# 第三遍：应用标签
salvaged = 0
new_chunks = []
for i, c in enumerate(chunks):
    labels = chunk_labels[i]
    extra = [t for t in salvage_map.get(i, []) if t not in labels][:MAX_SECONDARY]
    if extra:
        labels = labels + extra
        salvaged += 1
    new_chunks.append(rebuild(c, labels))

with open(KB, "w", encoding="utf-8") as f:
    for c in new_chunks:
        f.write(c + "\n")

print("「综合」重新归类：%d 块" % relabeled)
print("跨主题打捞（加了副标签）：%d 块" % salvaged)
print()
final = {}
for c in new_chunks:
    for t in label_of(c):
        final[t] = final.get(t, 0) + 1
print("增强后的主题覆盖（含副标签计数）：")
for t, n in sorted(final.items(), key=lambda x: -x[1]):
    mark = "  ← 打捞补厚" if counts.get(t, 0) < THIN_LIMIT else ""
    print("   %-16s %4d 块%s" % (t, n, mark))
print()
print("总块数不变：%d（打捞只是加标签，不复制内容）" % len(new_chunks))