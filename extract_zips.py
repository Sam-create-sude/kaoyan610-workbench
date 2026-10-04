"""解压资料包里的压缩文件（含包中包），提取到 data/source_extra/。

只读原始 zip，不改动 E:\610 考研 下的任何文件。
中文文件名需要特殊处理：Windows 压缩的 zip 常不带 UTF-8 标志，按 cp437 解出来会乱码，
需要重新用 GBK 解码。
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "source_extra")

ZIPS = [
    r"E:/610 考研/资料包/书本以外资料包(1).zip",
    r"E:/610 考研/资料包/设计与科技  0928(1).zip",
]


def fixed_name(info):
    name = info.filename
    if not (info.flag_bits & 0x800):
        try:
            name = name.encode("cp437").decode("gbk")
        except Exception:
            pass
    return name


def extract_one(zip_path, dest):
    os.makedirs(dest, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = fixed_name(info)
            if name.endswith("/") or name.endswith("\\"):
                continue
            target = os.path.join(dest, name.replace("/", os.sep).replace("\\", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                with z.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                n += 1
            except Exception as e:
                print("   跳过 %s (%s)" % (name, type(e).__name__))
    return n


def strip_ext(name):
    base = os.path.basename(name)
    return os.path.splitext(base)[0]


queue = [(p, os.path.join(OUT, strip_ext(p))) for p in ZIPS]
done = 0
rounds = 0

while queue:
    rounds += 1
    zpath, dest = queue.pop(0)
    if not os.path.exists(zpath):
        print("不存在：", zpath)
        continue
    print("[%d] 解压 %s" % (rounds, os.path.basename(zpath)))
    n = extract_one(zpath, dest)
    done += n
    print("    → %d 个文件" % n)

    # 找出刚解压出来的嵌套 zip
    for root, dirs, names in os.walk(dest):
        for nm in names:
            if nm.lower().endswith(".zip"):
                sub = os.path.join(root, nm)
                subdest = os.path.join(root, strip_ext(nm))
                queue.append((sub, subdest))

print()
print("解压完成，共释放 %d 个文件" % done)

stats = {}
for root, dirs, names in os.walk(OUT):
    for nm in names:
        ext = os.path.splitext(nm)[1].lower() or "(无扩展名)"
        stats[ext] = stats.get(ext, 0) + 1
print()
print("释放出的文件类型：")
for ext, c in sorted(stats.items(), key=lambda x: -x[1]):
    print("   %-10s %d 个" % (ext, c))
print()
print("输出目录：", OUT)