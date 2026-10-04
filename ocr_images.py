"""用视觉模型（PaddleOCR-VL）识别资料包里的图片，补齐传统 OCR 失去的内容。

比 tesseract 准得多——手写体、印刷体、表格里的中文都能正确还原。
输出并入 data/extracted/ 与 data/manifest.json，供 build_kb.py 使用。
"""
import os
import re
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

BASE = r"E:/610 考研/资料包/背诵资料(1)/背诵资料"
SOURCE_DIRS = [BASE, os.path.join(HERE, "data", "source_extra")]
DATA = os.path.join(HERE, "data")
EXTRACTED = os.path.join(DATA, "extracted")
MANIFEST = os.path.join(DATA, "manifest.json")
OCR_CACHE = os.path.join(DATA, "ocr_cache")
os.makedirs(OCR_CACHE, exist_ok=True)

client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"), base_url="https://api.siliconflow.cn/v1")
MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"

TOPIC_RULES = [
    ("碳中和", "碳中和与碳达峰"),
    ("碳达峰", "碳中和与碳达峰"),
    ("社会责任", "设计师的社会责任"),
    ("AIGC", "数智与AIGC"),
    ("数智", "数智与AIGC"),
    ("人工智能", "数智与AIGC"),
    ("教育", "设计教育"),
    ("教学", "设计教育"),
    ("数媒", "数媒与数字艺术"),
    ("数字媒体", "数媒与数字艺术"),
    ("科技", "设计与科技"),
    ("新技术", "设计与科技"),
    ("环境", "设计与环境"),
    ("生态", "设计与环境"),
    ("可持续", "设计与环境"),
    ("消费", "设计与消费"),
    ("品牌", "设计与消费"),
    ("创新", "设计与创新"),
    ("情感化", "情感化设计"),
    ("人性化", "人性化设计"),
    ("人本", "人性化设计"),
    ("管理", "设计与管理"),
    ("文化", "设计与文化"),
    ("非遗", "设计与文化"),
    ("史论", "设计史论"),
    ("设计学概论", "设计史论"),
    ("热点", "设计热点"),
]


def guess_topic(rel):
    for kw, topic in TOPIC_RULES:
        if kw in rel:
            return topic
    return "综合"


def safe_name(rel):
    name = rel.replace("\\", "__").replace("/", "__")
    name = re.sub(r"[^\w\u4e00-\u9fff\.\-]", "_", name)
    return name[:80]


def ocr_image(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = "png" if path.lower().endswith(".png") else "jpeg"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请把这张图片里的所有文字完整、准确地转录出来，保持原有的标题与条目层次。只输出转录的文字，不要添加任何解释或评论。"},
                {"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % (ext, b64)}},
            ],
        }],
    )
    return resp.choices[0].message.content or ""


def clean_ocr(text):
    text = re.sub(r"<\|LOC_\d+\|>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.search(r"公众号|小红书|甜柠檬|微信号|扫码|加我|关注", line):
            continue
        if len(line) < 4:
            continue
        out.append(line)
    return "\n".join(out)


with open(MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)
existing_sources = {m["source"] for m in manifest}
next_index = max(m["index"] for m in manifest) + 1

images = []
for base in SOURCE_DIRS:
    if not os.path.isdir(base):
        continue
    for root, dirs, names in os.walk(base):
        for n in names:
            if n.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, n)
                images.append((path, os.path.relpath(path, base)))
images.sort(key=lambda x: x[1])

print("找到 %d 张图片" % len(images))
added = 0
for path, rel in images:
    if rel in existing_sources:
        continue
    topic = guess_topic(rel)
    cache_file = os.path.join(OCR_CACHE, safe_name(rel) + ".txt")
    if os.path.exists(cache_file):
        text = open(cache_file, encoding="utf-8").read()
        print("  复用 OCR 缓存：%s（%d 字）" % (rel, len(text)))
    else:
        print("  识别 %s ..." % rel)
        try:
            raw = ocr_image(path)
        except Exception as e:
            print("     失败：%s %s" % (type(e).__name__, e))
            continue
        text = clean_ocr(raw)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
    if len(text) < 60:
        print("     内容过短(%d)，跳过" % len(text))
        continue
    out_name = "%03d_%s.txt" % (next_index, safe_name(rel))
    with open(os.path.join(EXTRACTED, out_name), "w", encoding="utf-8") as f:
        f.write("【主题】%s\n【来源】%s（图片OCR）\n\n%s\n" % (topic, rel, text))
    manifest.append({
        "index": next_index,
        "source": rel,
        "topic": topic,
        "chars": len(text),
        "file": out_name,
        "via": "vision-ocr",
    })
    next_index += 1
    added += 1
    print("     ✅ %d 字 → %s" % (len(text), out_name))

manifest.sort(key=lambda m: m["index"])
with open(MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print()
print("新增 %d 个图片资料，manifest 现有 %d 条" % (added, len(manifest)))