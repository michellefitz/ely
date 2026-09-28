"""Put every bottle image on one studio standard, without AI.

For each source image: find the bottle (alpha channel, or flood-fill of a
plain light background from the edges), crop to it, scale it to a fixed
height, and place it on a fixed backdrop with a floor shadow.

Images with a busy or dark background are not processed. They are listed
as "needs AI" in the report, because a cut-out would not be clean.

Usage: python3 normalise.py <raw_dir> <out_dir> <report.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 800, 1000            # 4:5 tile
FLOOR = 0.935               # bottle base line, fraction of H
MAX_H = 0.84                # bottle height, fraction of H
MAX_W = 0.78                # width cap for wide items (gift tubes, boxes)
TOP, BOTTOM = (234, 231, 227), (214, 209, 204)


def backdrop():
    y = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array(TOP), np.array(BOTTOM)
    grad = top + (bot - top) * y ** 1.4
    return Image.fromarray(np.repeat(grad, W, axis=1).astype("uint8"), "RGB")


def border_stats(rgb):
    """Median border colour, and the share of border pixels close to it."""
    a = np.asarray(rgb).astype(int)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    med = np.median(edge, axis=0).astype(int)
    share = (np.abs(edge - med).max(axis=1) <= 12).mean()
    return med, share


def mask_from_plain_background(rgb, tol):
    """Flood-fill near-background pixels from the image edges."""
    a = np.asarray(rgb).astype(int)
    bg, _ = border_stats(rgb)
    dist = np.abs(a - bg).max(axis=2)
    near = Image.fromarray(((dist <= tol) * 255).astype("uint8"), "L")
    # Pad so one seed reaches every edge pixel.
    pad = Image.new("L", (near.width + 2, near.height + 2), 255)
    pad.paste(near, (1, 1))
    ImageDraw.floodfill(pad, (0, 0), 128)
    fill = np.asarray(pad)[1:-1, 1:-1] == 128
    return ~fill


def fill_rows(mask):
    """Close small gaps across each row.

    A white label that touches the bottle edge lets the flood-fill leak in.
    Bottles are solid across each row, so a gap narrower than a fifth of the
    object width is part of the bottle. Wider gaps (a bottle beside its gift
    tube) stay open.
    """
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return mask
    max_gap = 0.2 * (cols[-1] - cols[0] + 1)
    out = mask.copy()
    for y in np.where(mask.any(axis=1))[0]:
        xs = np.where(mask[y])[0]
        gaps = np.where(np.diff(xs) > 1)[0]
        for g in gaps:
            a, b = xs[g], xs[g + 1]
            if b - a < max_gap:
                out[y, a:b] = True
    return out


def mirror_fill(mask):
    """Rebuild a bitten edge from the other side of the bottle.

    A white label that reaches the bottle edge is lost to the flood-fill on
    that side. A single bottle is symmetric, so each row is widened to the
    larger of its two half-widths around the bottle's centre line. Wide
    objects (a bottle beside its gift tube) are left alone.
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or (cols[-1] - cols[0]) > 0.5 * (rows[-1] - rows[0]):
        return mask
    lower = rows[len(rows) // 2:]
    mids = [(np.where(mask[y])[0][0] + np.where(mask[y])[0][-1]) / 2 for y in lower]
    cx = float(np.median(mids))
    out = mask.copy()
    for y in rows:
        xs = np.where(mask[y])[0]
        half = max(cx - xs[0], xs[-1] - cx)
        out[y, max(0, int(round(cx - half))):int(round(cx + half)) + 1] = True
    return out


def clean(mask):
    mask = mirror_fill(fill_rows(mask))
    m = Image.fromarray((mask * 255).astype("uint8"), "L")
    m = m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return m.filter(ImageFilter.GaussianBlur(0.8))


def classify(img):
    """Return (mask or None, reason)."""
    if img.mode in ("RGBA", "LA", "P") and "A" in img.convert("RGBA").getbands():
        alpha = np.asarray(img.convert("RGBA"))[:, :, 3]
        if (alpha < 16).mean() > 0.2:
            return clean(alpha > 16), "transparent"
    rgb = img.convert("RGB")
    med, share = border_stats(rgb)
    if med.min() > 225 and share > 0.8:
        # A cut-out that keeps most of the frame kept the backdrop too.
        # Retry looser, then give up.
        for tol in (18, 34):
            mask = mask_from_plain_background(rgb, tol)
            if mask.mean() < 0.45:
                return clean(mask), "white"
        return None, "backdrop not separable"
    return None, "busy background"


def touches_edge(mask):
    """True when the product runs off the top or bottom of the source."""
    a = np.asarray(mask) > 128
    return a[:2].any() or a[-2:].any()


def place(img, mask):
    box = mask.getbbox()
    if not box:
        return None
    cut = img.convert("RGB").crop(box)
    m = mask.crop(box)
    s = min(MAX_H * H / cut.height, MAX_W * W / cut.width)
    size = (max(1, round(cut.width * s)), max(1, round(cut.height * s)))
    cut, m = cut.resize(size, Image.LANCZOS), m.resize(size, Image.LANCZOS)
    out = backdrop()
    x, y = (W - size[0]) // 2, round(FLOOR * H) - size[1]
    # Floor shadow: soft ellipse under the base.
    sh = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(sh)
    rx, ry = size[0] * 0.62, max(6, H * 0.012)
    cy = y + size[1] - 2
    d.ellipse([W / 2 - rx, cy - ry, W / 2 + rx, cy + ry], fill=120)
    sh = sh.filter(ImageFilter.GaussianBlur(10))
    out.paste((58, 40, 44), (0, 0), sh)
    out.paste(cut, (x, y), m)
    return out


def main(raw_dir, out_dir, report_path):
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for f in sorted(raw.iterdir()):
        img = Image.open(f)
        src_w, src_h = img.size
        mask, reason = classify(img)
        entry = {"source": reason, "w": src_w, "h": src_h}
        if mask is not None:
            if touches_edge(mask):
                entry["cropped"] = True
            placed = place(img, mask)
            if placed is None:
                entry["status"] = "failed"
            else:
                placed.save(out / f"{f.name}.jpg", quality=84, optimize=True, progressive=True)
                entry["status"] = "normalised"
        else:
            entry["status"] = "needs AI"
        if max(src_w, src_h) < 600:
            entry["low_res"] = True
        report[f.name] = entry
    Path(report_path).write_text(json.dumps(report, indent=1))
    counts = {}
    for e in report.values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main(*sys.argv[1:4])
