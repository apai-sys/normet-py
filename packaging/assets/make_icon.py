#!/usr/bin/env python3
"""Generate the normet app icon (png/ico/icns) into packaging/assets/.

Run once locally (or whenever the design changes) and commit the results —
CI does not regenerate icons on every build. Needs Pillow; `iconutil` (macOS
only) is used for the .icns, with a graceful skip elsewhere. Rasterising an
SVG source (--svg) additionally needs PySide6 (already a GUI dependency).

Usage:
    python packaging/assets/make_icon.py --svg /path/to/logo.svg
    python packaging/assets/make_icon.py --letter N --color 2c7bb6   # no logo yet
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_icon(size: int, letter: str, color: tuple[int, int, int]) -> Image.Image:
    """Fallback placeholder: a solid circle badge with a letter."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(size * 0.06)
    d.ellipse([pad, pad, size - pad, size - pad], fill=(*color, 255))
    font = _font(int(size * 0.55))
    bbox = d.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(
        ((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]),
        letter,
        font=font,
        fill=(255, 255, 255, 255),
    )
    return img


def render_svg(svg_path: Path, size: int, margin: float = 0.04) -> Image.Image:
    """Rasterise an SVG onto a transparent square canvas, centred and scaled
    to fit (aspect ratio preserved) — the real logo, in place of a placeholder.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([sys.argv[0]])
    _ = app  # keep the (module-level singleton) app alive for the render call

    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise ValueError(f"invalid or unreadable SVG: {svg_path}")

    vbox = renderer.viewBoxF()
    if vbox.isEmpty():
        default = renderer.defaultSize()
        vbox = QRectF(0, 0, default.width(), default.height())

    avail = size * (1 - 2 * margin)
    scale = avail / max(vbox.width(), vbox.height())
    w, h = vbox.width() * scale, vbox.height() * scale
    x, y = (size - w) / 2, (size - h) / 2

    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(x, y, w, h))
    painter.end()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        image.save(str(tmp_path), "PNG")
        return Image.open(tmp_path).convert("RGBA").copy()
    finally:
        tmp_path.unlink(missing_ok=True)


def make_icns(png_1024: Path, out_icns: Path) -> bool:
    if sys.platform != "darwin" or shutil.which("iconutil") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        base = Image.open(png_1024)
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for s in sizes:
            base.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            if s <= 512:
                base.resize((s * 2, s * 2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out_icns)], check=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", type=Path, help="source logo to rasterise (preferred)")
    ap.add_argument("--letter", default="N", help="fallback placeholder letter")
    ap.add_argument("--color", default="2c7bb6", help="fallback placeholder colour, hex RGB no '#'")
    ap.add_argument("--name", default="normet")
    args = ap.parse_args()

    if args.svg:
        img = render_svg(args.svg, 1024)
    else:
        r, g, b = (int(args.color[i : i + 2], 16) for i in (0, 2, 4))
        img = draw_icon(1024, args.letter, (r, g, b))

    png_path = HERE / f"{args.name}.png"
    img.resize((256, 256), Image.LANCZOS).save(png_path)
    print(f"wrote {png_path}")

    ico_path = HERE / f"{args.name}.ico"
    img.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"wrote {ico_path}")

    icns_path = HERE / f"{args.name}.icns"
    tmp_1024 = HERE / f"_{args.name}_1024.png"
    img.save(tmp_1024)
    try:
        if make_icns(tmp_1024, icns_path):
            print(f"wrote {icns_path}")
        else:
            print("skipped .icns (iconutil not available on this platform)")
    finally:
        tmp_1024.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
