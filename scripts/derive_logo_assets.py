"""Derive docs-site logo assets from a high-resolution source image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = (
    ROOT / "logo" / "scDLKit logo.jpg",
    ROOT / "logo" / "logo.png",
)
STATIC_DIR = ROOT / "docs" / "_static"
WHITE_THRESHOLD = 245


def _resolve_source() -> Path:
    for candidate in SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate
    msg = "Missing source logo. Expected one of: " + ", ".join(
        str(path.relative_to(ROOT)) for path in SOURCE_CANDIDATES
    )
    raise FileNotFoundError(msg)


def _remove_white_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = []
    for red, green, blue, alpha in rgba.getdata():
        if alpha == 0:
            pixels.append((red, green, blue, alpha))
            continue
        if red >= WHITE_THRESHOLD and green >= WHITE_THRESHOLD and blue >= WHITE_THRESHOLD:
            pixels.append((red, green, blue, 0))
        else:
            pixels.append((red, green, blue, alpha))
    rgba.putdata(pixels)
    return rgba


def _crop_to_content(image: Image.Image, *, padding_ratio: float) -> Image.Image:
    rgba = _remove_white_background(image)
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda value: 255 if value > 0 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        msg = "No visible pixels found in source logo."
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
    source_path = _resolve_source()

    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    source = Image.open(source_path).convert("RGBA")
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
    print(f"- source: {source_path.relative_to(ROOT)}")
    for output in (
        STATIC_DIR / "scdlkit_logo.png",
        STATIC_DIR / "scdlkit_logo_mark.png",
        STATIC_DIR / "scdlkit_favicon.ico",
        STATIC_DIR / "scdlkit_favicon.png",
    ):
        print(f"- {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
