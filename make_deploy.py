"""生成精简发布目录 deploy/，供上云使用。

只复制运行必需的 6 个东西。
⚠️ 刻意不包含 .env —— 每人必须自带 API Key，服务端不持有任何密钥。
改完代码后重新跑一次本脚本即可同步。
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(HERE, "deploy")

# (源路径, 相对 deploy 的目标路径)
FILES = [
    ("kb_core.py", "kb_core.py"),
    ("server.py", "server.py"),
    ("web/index.html", "web/index.html"),
    ("web/pay-qr.jpg", "web/pay-qr.jpg"),
    ("data/knowledge_610.txt", "data/knowledge_610.txt"),
    ("data/knowledge_public.txt", "data/knowledge_public.txt"),
    ("data/vectors.json", "data/vectors.json"),
]

REQUIREMENTS = "openai>=1.0.0\n"


def main():
    if os.path.isdir(DEPLOY):
        shutil.rmtree(DEPLOY)
    os.makedirs(DEPLOY)

    for src_rel, dst_rel in FILES:
        src = os.path.join(HERE, src_rel.replace("/", os.sep))
        dst = os.path.join(DEPLOY, dst_rel.replace("/", os.sep))
        if not os.path.exists(src):
            print("  ⚠ 缺少 %s，跳过" % src_rel)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        print("  ✅ %-32s %8.1f KB" % (dst_rel, size / 1024))

    with open(os.path.join(DEPLOY, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS)
    print("  ✅ %-32s %8s" % ("requirements.txt", "24 B"))

    total = 0
    for root, dirs, names in os.walk(DEPLOY):
        for nme in names:
            total += os.path.getsize(os.path.join(root, nme))
    print()
    print("发布目录：%s" % DEPLOY)
    print("总体积：%.1f MB" % (total / 1024 / 1024))
    print()
    print("已排除：.env（含密钥）、data/extracted（原始文本）、")
    print("        data/source_extra（132MB 解压包）、data/ocr_cache、output/")
    print()
    print("✅ 发布包不含任何密钥——访客必须自带 Key，额度各自结算。")


if __name__ == "__main__":
    main()