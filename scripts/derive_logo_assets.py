"""Derive docs-site logo assets from a high-resolution source PNG."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logo" / "logo.png"
STATIC_DIR = ROOT / "docs" / "_static"


def _crop_to_content(image: Image.Image, *, padding_ratio: float) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda value: 255 if value > 0 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        msg = f"No visible pixels found in {SOURCE}."
        raise ValueError(msg)

    content = rgba.crop(bbox)
    width, height = content.size
    side = max(width, height)
    pad = max(8, int(side * padding_ratio))
    canvas = Image.new("RGBA", (side + 2 * pad, side + 2 * pad), (0, 0, 0, 0))
    offset = ((canvas.width - width) // 2, (canvas.height - height) // 2)
    canvas.paste(content, offset, content)
    return canvas


def _save_png(image: Image.Image, path: Path, size: int) -> None:
    resized = image.resize((size, size), Image.LANCZOS)
    resized.save(path)


def main() -> None:
    if not SOURCE.exists():
        msg = f"Missing source logo: {SOURCE}"
        raise FileNotFoundError(msg)

    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    source = Image.open(SOURCE).convert("RGBA")
    logo_canvas = _crop_to_content(source, padding_ratio=0.12)
    favicon_canvas = _crop_to_content(source, padding_ratio=0.18)

    _save_png(logo_canvas, STATIC_DIR / "scdlkit_logo.png", 512)
    _save_png(favicon_canvas, STATIC_DIR / "scdlkit_logo_mark.png", 256)

    favicon = favicon_canvas.resize((64, 64), Image.LANCZOS)
    favicon.save(
        STATIC_DIR / "scdlkit_favicon.ico",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
    )
    favicon.save(STATIC_DIR / "scdlkit_favicon.png")

    print("Derived logo assets:")
    for output in (
        STATIC_DIR / "scdlkit_logo.png",
        STATIC_DIR / "scdlkit_logo_mark.png",
        STATIC_DIR / "scdlkit_favicon.ico",
        STATIC_DIR / "scdlkit_favicon.png",
    ):
        print(f"- {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
