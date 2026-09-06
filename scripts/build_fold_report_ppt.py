#!/usr/bin/env python3
"""Reclassify the archive and build a simple white/black PPTX."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports" / "fold_run_archive_20260831"
RUNS = ROOT / "runs"
LOCAL_TZ = dt.timezone(dt.timedelta(hours=8))
CUTOFF = dt.datetime(2026, 8, 24, tzinfo=LOCAL_TZ)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        stderr=subprocess.DEVNULL, text=True, timeout=20,
    )
    return float(out.strip())


def choose_latest_longest() -> tuple[Path, float, dt.datetime]:
    candidates = []
    for path in RUNS.glob("**/*.mp4"):
        try:
            mt = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=LOCAL_TZ)
            if mt < CUTOFF:
                continue
            candidates.append((duration(path), mt, path))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("no recent rollout video found")
    def view_priority(path: Path) -> int:
        name = path.name.lower()
        if "composite" in name:
            return 3
        if "observer_rgb" in name or "camera_c" in name:
            return 2
        if name.endswith("_rgb.mp4"):
            return 1
        return 0
    # Prefer a human-readable RGB/composite view when durations are equal.
    dur, mt, path = sorted(candidates, key=lambda x: (x[0], view_priority(x[2]), x[1]), reverse=True)[0]
    return path, dur, mt


def make_latest_progress() -> dict:
    src, src_duration, src_mtime = choose_latest_longest()
    out_dir = ARCHIVE / "latest_progress"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "latest_progress_rollout_32x.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-vf", "setpts=PTS/32", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(out)
    ], check=True)
    poster = out_dir / "latest_progress_frame.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "1", "-i", str(out), "-frames:v", "1", str(poster)], check=True)
    meta = {
        "source": str(src),
        "source_mtime": src_mtime.isoformat(),
        "source_duration_s": round(src_duration, 3),
        "speed_factor": 32.0,
        "output": str(out),
        "output_duration_s": round(duration(out), 3),
        "selection": "longest available rollout in the recent seven-day window; latest timestamp breaks ties",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def reclassify_report() -> list[dict]:
    old = json.loads((ARCHIVE / "issue_catalog.json").read_text(encoding="utf-8"))
    structural = [x for x in old if x.get("class") == "structural"]
    planning = [
        {"id": "P1", "class": "planning", "title": "目标点语义选择不稳定", "archive_dir": "structural/S3_grounding_mask_mismatch", "finding": "Claude 有时把 sleeve 目标选到肩部、衣身或 Rxxx 内部点，即使 RGB 中袖子可见。需要部件拓扑、左右语义、外轮廓距离和反事实检查。", "evidence_summary": "S3 的 mask 对照、320 mm grounding 错位日志和 iteration 009 视频。"},
        {"id": "P2", "class": "planning", "title": "没有形成边缘优先抓取策略", "archive_dir": "structural/S1_acquisition_empty_grasp", "finding": "连续空抓后，Claude 仍主要改变高度、yaw 或小范围位移，没有稳定切换到 sleeve hem/free-boundary、edge-straddle 或 opposition。", "evidence_summary": "S1 的 Camera-C hold-check、telemetry 和 32 倍速 rollout。"},
        {"id": "P3", "class": "planning", "title": "运输与落放规划不足", "archive_dir": "structural/S2_whole_bundle_drag_roll", "finding": "有些迭代确实抓到布，但计划是零横向位移、回到原点，或把整团布拖成更窄更高的 roll。需要显式验证 source→destination 位移和落放后的轮廓扩展。", "evidence_summary": "S2 的 before/after、trajectory 和 AB 视频。"},
        {"id": "P4", "class": "planning", "title": "失败归因和探索多样性不足", "archive_dir": "structural/S7_unattended_state_loop", "finding": "多次失败后仍重复相近假设，并在 BUNCHED/left_sleeve 状态循环。需要维护已证伪假设集合，强制下一轮改变因果维度。", "evidence_summary": "S7 summary/debug.log、iteration 009 视频和保留的 experience ledger。"},
        {"id": "P5", "class": "planning", "title": "计划输出契约不稳定", "archive_dir": "planning/P5_plan_contract", "finding": "Claude 偶发缺少 safety_notes、close_gripper、move target 或返回非法 skill invocation。需要更短的 schema 提示、规划前自检和局部重试。", "evidence_summary": "P5 目录中的 schema/contract rejection 日志。"},
    ]
    operational = [
        {"id": "O1", "class": "operational", "title": "网络、相机和 SDK 瞬时故障", "archive_dir": "operational/O1_infra_transient", "finding": "connect socket failed、API ENOTFOUND、SDK TypeError 属于运行基础设施问题，不计入 Claude planning 能力评估。", "evidence_summary": "O1 目录中的 socket、DNS 和 TypeError 日志。"},
        {"id": "O2", "class": "operational", "title": "人工中断和旧版运行上限", "archive_dir": "operational/O2_interruptions_limits", "finding": "INTERRUPTED、MAX_FOLDS_REACHED 和 KeyboardInterrupt 主要反映旧运行方式或人工停止。", "evidence_summary": "O2 目录中的 interrupted/max-fold summary。"},
    ]
    for src, dst in [
        (ARCHIVE / "nonstructural" / "N1_schema_contract", ARCHIVE / "planning" / "P5_plan_contract"),
        (ARCHIVE / "nonstructural" / "N2_infra_transient", ARCHIVE / "operational" / "O1_infra_transient"),
        (ARCHIVE / "nonstructural" / "N3_interruptions_limits", ARCHIVE / "operational" / "O2_interruptions_limits"),
    ]:
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    issues = structural + planning + operational
    (ARCHIVE / "issue_catalog.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    inv = json.loads((ARCHIVE / "inventory_meta.json").read_text(encoding="utf-8"))
    ev = json.loads((ARCHIVE / "evaluation_stats.json").read_text(encoding="utf-8"))
    timing = json.loads((ARCHIVE / "timing_summary.json").read_text(encoding="utf-8"))
    def section(items):
        chunks=[]
        for x in items:
            ad=x["archive_dir"]; n=len(list((ARCHIVE/ad).glob("*_32x.mp4")))
            video=f"视频证据：已归档 {n} 个 32 倍速视频。" if n else "视频证据：无（执行前失败，未产生 rollout 视频）。"
            chunks.append(f"### {x['id']} · {x['title']}\n\n{x['finding']}\n\n证据：{x['evidence_summary']}\n\n{video}\n\n归档目录：`{ad}`")
        return "\n\n".join(chunks)
    status_lines="\n".join(f"- `{k}`: {v}" for k,v in sorted(inv["summary_status_counts"].items()))
    eval_lines="\n".join(f"- `{k}`: {v}" for k,v in sorted(ev["counts"].items()))
    timing_lines="\n".join(f"- `{k}`: n={v['n']}, mean={v['mean_s']} s, median={v['median_s']} s, max={v['max_s']} s" for k,v in timing.items())
    report=f"""# 折叠衣服实验归档报告

统计窗口：2026-08-24 00:00（Asia/Shanghai）至 2026-08-31。

## 总览

- 顶层 run：**{inv['top_level_runs']}**
- summary 记录：**{sum(inv['summary_status_counts'].values())}**
- 有 evaluation 的迭代：**{ev['evaluated_iterations']}**
- 无人值守重复尝试不是独立样本。

Summary 状态分布：

{status_lines}

Evaluation 状态分布：

{eval_lines}

## 结构性问题

结构性问题是跨 run 重复出现、需要修改 perception、pipeline、控制约束或观测架构的问题。

{section(structural)}

## Claude planning 能力不足（原“非结构性问题”）

这里的非结构性问题专指 Claude 规划本身暴露出的算法缺口，需要通过候选排序、因果归因、探索策略和输出自检提高成功率；网络、SDK 和人工中断不计入此类。

{section(planning)}

## 运行性附录

{section(operational)}

## 时间开销

{timing_lines}

视觉 planning 和 final grounding 的耗时主要来自上下文增长、reselection/retry 和结构化输出重试，不是机械臂运动耗时。

## 归档说明

- 已复制并校验代表性图像、JSON、日志和视频证据。
- 视频证据为 32 倍速版本。
- 选中的源证据已删除；experience ledger 和未选中的原始数据保留。
"""
    (ARCHIVE / "report.md").write_text(report, encoding="utf-8")
    return issues


def fnt(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size=size, index=0)


def make_slide(title: str, bullets: list[str] | None = None, images: list[Path] | None = None, footer: str = "") -> Image.Image:
    W,H=1600,900; im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
    d.text((70,45),title,font=fnt(42,True),fill="black"); d.line((70,110,1530,110),fill="black",width=2)
    y=145
    for b in bullets or []:
        txt="\n".join(textwrap.wrap("• "+b,width=50,break_long_words=False,break_on_hyphens=False))
        d.multiline_text((90,y),txt,font=fnt(28),fill="black",spacing=10); y += 48*txt.count("\n")+52
    existing=[p for p in (images or []) if p.exists()]
    if existing:
        ax,ay,aw,ah=820,145,700,650; n=min(2,len(existing)); ew=(aw-20)//2 if n==2 else aw
        for i,p in enumerate(existing[:2]):
            pic=Image.open(p).convert("RGB"); pic.thumbnail((ew,ah)); x=ax+i*(ew+20) if n==2 else ax+(aw-pic.width)//2; yy=ay+(ah-pic.height)//2; im.paste(pic,(x,yy)); d.rectangle((x,yy,x+pic.width,yy+pic.height),outline="black",width=2)
    if footer: d.text((70,850),"\n".join(textwrap.wrap(footer,width=120)),font=fnt(18),fill="black")
    return im


def pptx_from_images(images: list[Path], out_path: Path) -> None:
    """Create an image-only PPTX.  Image slides keep the requested simple style."""
    ns_a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    ns_p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ct = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    for i in range(1, len(images) + 1):
        ct.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    ct.append('</Types>')
    sld_ids = ''.join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(len(images)))
    pres = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="{ns_a}" xmlns:r="{ns_r}" xmlns:p="{ns_p}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(images)+1}"/></p:sldMasterIdLst><p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'''
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(len(images)):
        rels.append(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>')
    rels.append(f'<Relationship Id="rId{len(images)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/></Relationships>')
    master = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="{ns_a}" xmlns:r="{ns_r}" xmlns:p="{ns_p}"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles/></p:sldMaster>'''
    layout = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="{ns_a}" xmlns:r="{ns_r}" xmlns:p="{ns_p}" type="blank"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'''
    theme = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="{ns_a}" name="Simple"><a:themeElements><a:clrScheme name="Simple"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="000000"/></a:dk2><a:lt2><a:srgbClr val="FFFFFF"/></a:lt2><a:accent1><a:srgbClr val="000000"/></a:accent1><a:accent2><a:srgbClr val="000000"/></a:accent2><a:accent3><a:srgbClr val="000000"/></a:accent3><a:accent4><a:srgbClr val="000000"/></a:accent4><a:accent5><a:srgbClr val="000000"/></a:accent5><a:accent6><a:srgbClr val="000000"/></a:accent6><a:hlink><a:srgbClr val="000000"/></a:hlink><a:folHlink><a:srgbClr val="000000"/></a:folHlink></a:clrScheme><a:fontScheme name="Simple"><a:majorFont/><a:minorFont/></a:fontScheme><a:fmtScheme name="Simple"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>'''
    root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>'
    core = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Fold experiment review</dc:title><dc:creator>clothAgent</dc:creator></cp:coreProperties>'
    app = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>clothAgent</Application><PresentationFormat>Widescreen</PresentationFormat></Properties>'
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        ct.insert(-1, '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>')
        ct.insert(-1, '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>')
        z.writestr("[Content_Types].xml", ''.join(ct)); z.writestr("_rels/.rels", root_rels); z.writestr("docProps/core.xml", core); z.writestr("docProps/app.xml", app); z.writestr("ppt/presentation.xml", pres); z.writestr("ppt/_rels/presentation.xml.rels", ''.join(rels)); z.writestr("ppt/slideMasters/slideMaster1.xml", master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>')
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout); z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>'); z.writestr("ppt/theme/theme1.xml", theme)
        for i, image in enumerate(images, 1):
            slide = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:a="{ns_a}" xmlns:r="{ns_r}" xmlns:p="{ns_p}"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/><p:pic><p:nvPicPr><p:cNvPr id="2" name="Picture {i}"/><p:cNvPicPr preferRelativeResize="0"/><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:ln/></p:spPr></p:pic></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'''
            rel = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image{i}.png"/></Relationships>'
            z.writestr(f"ppt/slides/slide{i}.xml", slide); z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rel); z.write(image, f"ppt/media/image{i}.png")


def pptx_from_template(images: list[Path], out_path: Path) -> None:
    """Reuse the repository's known-good image-slide PPTX template."""
    import re
    template = ROOT / "results" / "canonical_collar_label_garment" / "dino_experiment_presentation_16x9.pptx"
    if not template.exists():
        return pptx_from_images(images, out_path)
    with zipfile.ZipFile(template, "r") as zin:
        entries = {name: zin.read(name) for name in zin.namelist()}
    slide_xml = entries["ppt/slides/slide1.xml"]
    slide_rel_template = entries["ppt/slides/_rels/slide1.xml.rels"].decode("utf-8")
    # Keep all template infrastructure, replacing only slide/media metadata.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            if name in {"ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "[Content_Types].xml"} or name.startswith("ppt/slides/slide") or name.startswith("ppt/slides/_rels/") or name.startswith("ppt/media/"):
                continue
            z.writestr(name, data)
        pres = entries["ppt/presentation.xml"].decode("utf-8")
        pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", "<p:sldIdLst>" + "".join(f'<p:sldId id="{256+i}" r:id="rId{i+8}"/>' for i in range(len(images))) + "</p:sldIdLst>", pres)
        z.writestr("ppt/presentation.xml", pres.encode("utf-8"))
        rels = entries["ppt/_rels/presentation.xml.rels"].decode("utf-8")
        rels = re.sub(r'<Relationship Id="rId[3-9][0-9]*" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide[0-9]+.xml"/>', "", rels)
        rels = rels.replace("</Relationships>", "".join(f'<Relationship Id="rId{i+8}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>' for i in range(len(images))) + "</Relationships>")
        z.writestr("ppt/_rels/presentation.xml.rels", rels.encode("utf-8"))
        ct = entries["[Content_Types].xml"].decode("utf-8")
        ct = re.sub(r'<Override PartName="/ppt/slides/slide[0-9]+.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>', "", ct)
        ct = re.sub(r'<Override PartName="/ppt/slides/_rels/slide[0-9]+.xml.rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>', "", ct)
        ct = re.sub(r'<Override PartName="/ppt/media/image[0-9]+.png" ContentType="image/png"/>', "", ct)
        overrides = "".join(f'<Override PartName="/ppt/slides/slide{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/slides/_rels/slide{i+1}.xml.rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/ppt/media/image{i+1}.png" ContentType="image/png"/>' for i in range(len(images)))
        ct = ct.replace("</Types>", overrides + "</Types>")
        z.writestr("[Content_Types].xml", ct.encode("utf-8"))
        for i, image in enumerate(images, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml)
            rel = re.sub(r"image1.png", f"image{i}.png", slide_rel_template)
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rel.encode("utf-8"))
            z.write(image, f"ppt/media/image{i}.png")


def pptx_from_images_uno(images: list[Path], out_path: Path) -> None:
    """Use LibreOffice UNO to create a standards-compliant image-slide PPTX."""
    import subprocess as _subprocess
    import time as _time
    import uno
    from com.sun.star.awt import Point, Size
    from com.sun.star.beans import PropertyValue
    profile = ARCHIVE / "uno_profile"
    profile.mkdir(exist_ok=True)
    proc = _subprocess.Popen([
        "soffice", "--headless", "--norestore", "--nofirststartwizard",
        f"-env:UserInstallation=file://{profile}",
        "--accept=socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext",
    ], stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        ctx = None
        for _ in range(60):
            try:
                ctx = resolver.resolve("uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
                break
            except Exception:
                _time.sleep(0.2)
        if ctx is None:
            raise RuntimeError("could not connect to LibreOffice UNO")
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
        pages = doc.getDrawPages()
        # Default Impress documents start with one blank page.
        while pages.getCount() > 1:
            pages.remove(pages.getByIndex(pages.getCount() - 1))
        page_w, page_h = 28000, 15750
        for idx, image_path in enumerate(images):
            page = pages.getByIndex(0) if idx == 0 else pages.insertNewByIndex(idx)
            page.Width = page_w; page.Height = page_h
            shape = doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
            shape.Position = Point(0, 0); shape.Size = Size(page_w, page_h)
            shape.GraphicURL = uno.systemPathToFileUrl(str(image_path))
            page.add(shape)
        url = uno.systemPathToFileUrl(str(out_path))
        props = (PropertyValue(Name="FilterName", Value="Impress MS PowerPoint 2007 XML"), PropertyValue(Name="Overwrite", Value=True))
        doc.storeAsURL(url, props)
        doc.close(True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def build_ppt(latest: dict) -> Path:
    s = ARCHIVE / "structural"; lp = ARCHIVE / "latest_progress"
    slides = [
        make_slide("T-shirt 折叠实验：最近 7 天复盘", ["84 个顶层 run，288 条 summary，97 个有 evaluation 的迭代。", "83 次 acquisition failure；只有 5 次 acquisition 成功。", "白底黑字，附代表性图像和 32 倍速视频索引。"], footer="统计窗口：2026-08-24 至 2026-08-31（Asia/Shanghai）"),
        make_slide("结构性问题：系统与感知链路", ["袖子 acquisition 反复空抓。", "RGB→Rxxx→XYZ、mask 和可见性判定不一致。", "深度/桌面/海绵/IK/工作空间约束没有稳定统一。", "相机曝光和 table-luma gate 会在动作前阻塞。"], footer="报告 S1、S3、S5、S8。"),
        make_slide("Claude planning 能力不足（原“非结构性问题”）", ["目标点有时落到肩部/衣身，而不是袖口外缘。", "空抓后主要改变 Z/yaw，没有切换接触拓扑。", "抓到布后可能零横向位移或回到原点。", "失败归因和探索多样性不足，容易循环。", "输出 schema 偶发缺字段，需要规划前自检。"], footer="需要通过候选排序、因果账本和实验策略提高成功率。"),
        make_slide("证据 1：袖子空抓", ["Camera-C hold-check 没有悬挂布料。", "position_pulse=0；mechanical_grasp_detected=false。", "before/after 轮廓基本不变。"], [s/"S1_acquisition_empty_grasp"/"before_camera_A.png", s/"S1_acquisition_empty_grasp"/"hold_check_camera_C.png"], "视频：S1_acquisition_empty_grasp/camera_C_32x.mp4"),
        make_slide("证据 2：整团布拖动并滚卷", ["确实抓到布，但没有分离目标层。", "after 变成更窄、更高的 elongated roll。", "task_progress=REGRESSED；laydown=FAILURE。"], [s/"S2_whole_bundle_drag_roll"/"before_camera_B.png", s/"S2_whole_bundle_drag_roll"/"after_camera_B.png"], "视频：S2_whole_bundle_drag_roll/composite_AB_32x.mp4"),
        make_slide("证据 3：grounding / mask 不一致", ["selected point 与 production mask 不一致。", "另一次日志中 selected R031 与实际 grasp XY 相差约 320 mm。", "目标名为 left_sleeve，但实际落到肩部。"], [s/"S3_grounding_mask_mismatch"/"selected_point_vs_mask.png", s/"S3_grounding_mask_mismatch"/"night_iter009_after_A.png"], "视频：S3_grounding_mask_mismatch/night_iter009_camera_C_32x.mp4"),
        make_slide("证据 4：规划耗时", ["visual planning 平均 193.4 s，最大 1303.9 s。", "final grounding 平均 189.1 s，最大 391.8 s。", "上下文 rollover、reselection/retry 和结构化输出重试是主要来源。"], footer="timing_summary.json 位于归档根目录。"),
        make_slide("证据 5：观测和无人值守循环", ["Camera-A 被夹爪遮挡；Camera-C 是未标定 observer。", "telemetry 多次 simulated=true，只能弱 corroboration。", "长 run 仍停在 BUNCHED/left_sleeve。"], [s/"S6_observability_telemetry"/"night_hold_check.png", lp/"latest_progress_frame.png"], "视频：S6_observability_telemetry/night_camC_rollout_32x.mp4"),
        make_slide("最新进展：最长 rollout（32 倍速）", [f"来源：{latest['source']}", f"原始时长：{latest['source_duration_s']} s；32 倍速后约 {latest['output_duration_s']} s。", "文件：latest_progress/latest_progress_rollout_32x.mp4。", "该视频是最近 7 天窗口中可获得的最长 rollout，不等于任务成功。"], [lp/"latest_progress_frame.png"], "完整路径见 latest_progress/metadata.json。"),
    ]
    slide_dir = ARCHIVE / "ppt_slides"; slide_dir.mkdir(exist_ok=True)
    slide_paths = []
    for i, image in enumerate(slides, 1):
        path = slide_dir / f"slide_{i:02d}.png"; image.save(path); slide_paths.append(path)
    out = ARCHIVE / "fold_experiment_review.pptx"; pptx_from_template(slide_paths, out); return out


def main() -> int:
    issues = reclassify_report(); latest = make_latest_progress(); ppt = build_ppt(latest)
    (ARCHIVE / "latest_progress" / "ppt_manifest.json").write_text(json.dumps({"ppt": str(ppt), "slide_count": 9, "planning_issue_ids": [x["id"] for x in issues if x["class"] == "planning"], "latest_progress": latest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ppt": str(ppt), "latest_progress": latest, "planning_issue_ids": [x["id"] for x in issues if x["class"] == "planning"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
