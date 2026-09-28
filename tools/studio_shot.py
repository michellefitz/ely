"""Make a studio shot of one bottle with Nano Banana, then check the label.

Needs GEMINI_API_KEY in the environment, and network access to
generativelanguage.googleapis.com.

Step 1 sends the source photo and one reference studio shot to the image
model and asks for the same bottle on the ELY studio standard.
Step 2 sends the new image to a text model and asks it to read the label.
The script compares producer, wine name and vintage with what you pass in,
and prints PASS or REVIEW. A REVIEW result must go to a person.

Usage:
  python3 studio_shot.py SOURCE.jpg REFERENCE.jpg OUT.jpg \
      --producer "Le Piane" --wine "Boca" --vintage 2018
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
IMAGE_MODEL = os.environ.get("ELY_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
CHECK_MODEL = os.environ.get("ELY_CHECK_MODEL", "gemini-2.5-flash")

PROMPT = """You are a product photographer for a wine shop.
Image 1 is a photo of a real bottle. Image 2 shows the house studio style.
Make a new photo of the bottle in image 1, in the style of image 2:
- Same bottle. Keep the label, text, vintage, capsule, glass colour and
  bottle shape exactly as in image 1. Do not add, remove or change any text.
- Bottle upright, centred, front label square to the camera.
- Plain warm light-grey studio backdrop, soft top-left key light,
  gentle rim light on the glass, soft floor shadow.
- Bottle fills about 84% of the frame height, base near the bottom.
- 4:5 portrait. No props, no hands, no reflections of the room."""

CHECK = """Read the front label of the wine bottle in this image.
Reply with JSON only: {"producer": "", "wine": "", "vintage": "", "other_text": ""}.
Use "" for anything you cannot read. Do not guess."""


def part(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.b64encode(open(path, "rb").read()).decode()
    return {"inline_data": {"mime_type": mime, "data": data}}


def call(model, parts, config):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set.")
    body = json.dumps({"contents": [{"parts": parts}], "generationConfig": config}).encode()
    req = urllib.request.Request(API.format(model=model), data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def generate(source, reference, out):
    res = call(IMAGE_MODEL, [{"text": PROMPT}, part(source), part(reference)],
               {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "4:5"}})
    for p in res["candidates"][0]["content"]["parts"]:
        blob = p.get("inlineData") or p.get("inline_data")
        if blob:
            open(out, "wb").write(base64.b64decode(blob["data"]))
            return
    sys.exit("No image in the response: " + json.dumps(res)[:500])


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def check(out, expected):
    res = call(CHECK_MODEL, [{"text": CHECK}, part(out)], {"responseMimeType": "application/json"})
    read = json.loads(res["candidates"][0]["content"]["parts"][0]["text"])
    problems = [f"{k}: expected {v!r}, label reads {read.get(k, '')!r}"
                for k, v in expected.items() if v and norm(v) not in norm(read.get(k, ""))]
    return read, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("reference"); ap.add_argument("out")
    ap.add_argument("--producer", default=""); ap.add_argument("--wine", default="")
    ap.add_argument("--vintage", default="")
    ap.add_argument("--tries", type=int, default=2)
    a = ap.parse_args()
    expected = {"producer": a.producer, "wine": a.wine, "vintage": a.vintage}
    for n in range(1, a.tries + 1):
        generate(a.source, a.reference, a.out)
        read, problems = check(a.out, expected)
        print(json.dumps({"try": n, "label": read, "problems": problems}, ensure_ascii=False))
        if not problems:
            print("PASS", a.out)
            return
    print("REVIEW", a.out, "- a person must check this image before it is used.")


if __name__ == "__main__":
    main()
