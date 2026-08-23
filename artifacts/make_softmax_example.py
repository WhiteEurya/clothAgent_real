from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont, ImageFilter


OUT = Path(__file__).with_name("softmax_vs_local_contrast_example.png")
W, H = 1600, 900


def font(size: int, bold: bool = False):
    path = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if bold
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def lerp(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def heat_color(v: float):
    v = max(0.0, min(1.0, v))
    stops = [(0.0, (24, 55, 130)), (0.38, (22, 153, 205)),
             (0.68, (250, 220, 65)), (1.0, (214, 50, 40))]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if v <= x1:
            return lerp(c0, c1, (v - x0) / (x1 - x0))
    return stops[-1][1]


def draw_panel(base: Image.Image, x0: int, title: str, mode: str):
    draw = ImageDraw.Draw(base)
    panel = (x0, 105, x0 + 700, 790)
    draw.rounded_rectangle(panel, radius=28, fill=(247, 249, 252), outline=(210, 216, 226), width=3)
    draw.text((x0 + 34, 130), title, fill=(25, 33, 48), font=font(34, bold=True))

    ox, oy, pw, ph = x0 + 66, 215, 568, 430
    mask = Image.new("L", (pw, ph), 0)
    md = ImageDraw.Draw(mask)
    poly = [(56, 80), (168, 28), (350, 48), (505, 116), (530, 285),
            (427, 384), (245, 410), (85, 340), (25, 215)]
    md.polygon(poly, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))

    field = Image.new("RGB", (pw, ph), (235, 239, 247))
    px = field.load()
    for y in range(ph):
        for x in range(pw):
            broad = 0.18 * math.exp(-(((x - 300) / 240) ** 2 + ((y - 240) / 200) ** 2))
            small = 0.23 * math.exp(-(((x - 235) / 34) ** 2 + ((y - 215) / 30) ** 2))
            extreme = 1.0 * math.exp(-(((x - 430) / 20) ** 2 + ((y - 125) / 20) ** 2))
            raw = 0.08 + broad + small + extreme
            if mode == "softmax":
                v = math.exp(8.0 * raw)
                v = (v - math.exp(8.0 * 0.08)) / (math.exp(8.0 * 1.26) - math.exp(8.0 * 0.08))
                v = max(0.0, min(1.0, v))
            else:
                v = max(0.0, min(1.0, raw / 1.26))
            px[x, y] = heat_color(v)

    base.paste(field, (ox, oy), mask)
    shifted = [(ox + x, oy + y) for x, y in poly]
    draw.line(shifted + [shifted[0]], fill=(44, 53, 75), width=4, joint="curve")

    if mode == "multiscale":
        overlay = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        cx, cy = ox + 235, oy + 215
        for r, alpha in [(58, 25), (48, 42), (38, 70), (29, 125)]:
            od.ellipse((cx-r, cy-r*0.8, cx+r, cy+r*0.8), outline=(0, 232, 238, alpha), width=5)
        base.alpha_composite(overlay)
        draw = ImageDraw.Draw(base)
        draw.ellipse((ox + 410, oy + 105, ox + 450, oy + 145), outline=(255, 255, 255), width=7)
        draw.ellipse((ox + 416, oy + 111, ox + 444, oy + 139), outline=(255, 222, 38), width=4)
        draw.text((x0 + 60, 670), "局部残差：青色轮廓保留小凸起", fill=(0, 126, 142), font=font(24, bold=True))
    else:
        draw = ImageDraw.Draw(base)
        draw.ellipse((ox + 410, oy + 105, ox + 450, oy + 145), outline=(255, 255, 255), width=7)
        draw.ellipse((ox + 416, oy + 111, ox + 444, oy + 139), outline=(255, 222, 38), width=4)
        draw.text((x0 + 60, 670), "极值主导：其他高度几乎被压平", fill=(157, 55, 55), font=font(24, bold=True))

    if mode == "softmax":
        draw.line((ox + 235, oy + 200, x0 + 470, 580), fill=(93, 102, 120), width=3)
        draw.rounded_rectangle((x0 + 425, 525, x0 + 650, 610), radius=15, fill=(255, 255, 255), outline=(195, 201, 212), width=2)
        draw.text((x0 + 445, 540), "小高度差\n几乎不可见", fill=(55, 63, 80), font=font(23, bold=True), spacing=5)
    else:
        draw.line((ox + 235, oy + 200, x0 + 480, 580), fill=(0, 176, 187), width=3)
        draw.rounded_rectangle((x0 + 425, 525, x0 + 650, 610), radius=15, fill=(235, 255, 255), outline=(0, 188, 198), width=2)
        draw.text((x0 + 445, 540), "小高度差\n仍然清晰", fill=(0, 108, 120), font=font(23, bold=True), spacing=5)


img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((60, 35), "高度图可视化示例：全局极值 + 局部高度变化", fill=(18, 27, 43), font=font(38, bold=True))
draw.text((62, 80), "同一个极值点存在时，比较 softmax 与多尺度编码", fill=(88, 98, 116), font=font(22))
draw_panel(img, 60, "全局 softmax", "softmax")
draw_panel(img, 840, "全局高度 + 局部残差", "multiscale")
draw = ImageDraw.Draw(img)
draw.rounded_rectangle((260, 815, 1340, 870), radius=18, fill=(245, 247, 251), outline=(218, 223, 231), width=2)
legend = [("低", (24, 55, 130)), ("中", (22, 153, 205)), ("高", (250, 220, 65)), ("全局极值", (214, 50, 40)), ("局部残差", (0, 210, 220))]
lx = 295
for label, color in legend:
    draw.rounded_rectangle((lx, 830, lx + 26, 854), radius=5, fill=color)
    draw.text((lx + 36, 827), label, fill=(60, 69, 84), font=font(18))
    lx += 190 if label != "全局极值" else 220
img.convert("RGB").save(OUT, quality=95)
print(OUT)
