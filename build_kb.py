"""把 data/extracted/ 下的纯文本拼装成知识库 knowledge_610.txt。

切分原则（用上午学到的结论）：
  1. 在【单个文档内部】把相邻行合并到接近 MAX_LEN —— 保住语义完整
  2. 【绝不跨文档合并】—— 每个文档是一个话题边界
  3. 每块前面贴【主题】标签 —— 既提升检索命中，也让答案能给出处
"""
import os
import re
import json

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EXTRACTED = os.path.join(DATA, "extracted")
MANIFEST = os.path.join(DATA, "manifest.json")
OUT = os.path.join(DATA, "knowledge_610.txt")

MAX_LEN = 400


def join_two(a, b):
    if a and b and a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
        return a + " " + b
    return a + b


with open(MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)

all_chunks = []


def expand_long_lines(lines):
    """长度超限的行，先按句末标点拆开，避免出现上千字的巨块。"""
    out = []
    for line in lines:
        if len(line) <= MAX_LEN:
            out.append(line)
            continue
        tmp = line.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n")
        out.extend([p.strip() for p in tmp.split("\n") if p.strip()])
    return out


def cn_ratio(s):
    """中文字符占比。低于阈值的基本是英文摘要/关键词行，对中文论述无用。"""
    if not s:
        return 0.0
    cn = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return cn / len(s)


def is_junk(s):
    """判断是不是垃圾块。四类噪声：字体映射失败的高重复串、英文摘要、
    关键词行、参考文献列表。"""
    if not s or len(s) < 20:
        return True
    # ① 中文字符占比过低
    if cn_ratio(s) < 0.35:
        return True
    # ② 单一字符重复过多（如 mamamamama、▃▃▃▃）
    top = max(s.count(ch) for ch in set(s))
    if top / len(s) > 0.3:
        return True
    # ③ 以英文摘要 / 关键词 / 参考文献开头
    head = s.lstrip()[:20]
    if re.match(r"(?i)^(key\s?words?|abstract|references|摘要[:：]?\s*$)", head):
        return True
    if head.startswith("参考文献") or head.startswith("References"):
        return True
    # ④ 引用编号密集（参考文献列表特征）
    if len(re.findall(r"\[\d+\]", s)) >= 4:
        return True
    return False


def flush(buf, topic, source):
    """兜底：万一还有超长块，按 MAX_LEN 硬切。"""
    while len(buf) > MAX_LEN:
        all_chunks.append(("【%s】%s" % (topic, buf[:MAX_LEN]), source))
        buf = buf[MAX_LEN:]
    # 清掉 NUL 字节与解码残留的替换字符，避免污染知识库
    buf = buf.replace("\x00", "").replace("\ufffd", "")
    if buf and not is_junk(buf):
        all_chunks.append(("【%s】%s" % (topic, buf), source))


for item in manifest:
    path = os.path.join(EXTRACTED, item["file"])
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = [l.strip() for l in raw.split("\n")]
    body = [l for l in lines if l and not l.startswith("【")]
    if not body:
        continue
    body = expand_long_lines(body)

    # 文档内部合并，绝不跨文档
    buf = ""
    for line in body:
        if len(buf) + len(line) <= MAX_LEN:
            buf = join_two(buf, line)
        else:
            if buf:
                flush(buf, item["topic"], item["source"])
            buf = line
    if buf:
        flush(buf, item["topic"], item["source"])

with open(OUT, "w", encoding="utf-8") as f:
    for chunk, src in all_chunks:
        f.write(chunk.replace("\n", " ") + "\n")

lengths = [len(c) for c, s in all_chunks]
print("=" * 62)
print("知识库已生成：", OUT)
print("总块数：%d" % len(all_chunks))
print("总字数：%d" % sum(lengths))
print("平均每块：%d 字" % (sum(lengths) // len(lengths)))
print("最短 %d 字 / 最长 %d 字" % (min(lengths), max(lengths)))
print()
by_topic = {}
for c, s in all_chunks:
    t = c[1:c.index("】")]
    by_topic[t] = by_topic.get(t, 0) + 1
print("各主题块数：")
for t, n in sorted(by_topic.items(), key=lambda x: -x[1]):
    print("   %-14s %3d 块" % (t, n))
print()
print("前 3 块预览：")
for c, s in all_chunks[:3]:
    print("   ", c[:88], "...")