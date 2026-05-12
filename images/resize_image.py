#!/usr/bin/env python3
"""
Redimensionne une image à 1738x1333 sans déformation.
L'image est mise à l'échelle proportionnellement puis centrée avec des bords blancs.
"""

from PIL import Image
import sys
import os

TARGET_W, TARGET_H = 1738, 1333

def resize_with_padding(input_path: str, output_path: str = None):
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_resized{ext}"

    img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img.size

    # Calcul du ratio pour tenir dans 1738x1333 sans déformer
    ratio = min(TARGET_W / orig_w, TARGET_H / orig_h)
    new_w = round(orig_w * ratio)
    new_h = round(orig_h * ratio)

    # Redimensionnement proportionnel
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Création du canvas blanc 1738x1333
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))

    # Centrage de l'image sur le canvas
    offset_x = (TARGET_W - new_w) // 2
    offset_y = (TARGET_H - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    canvas.save(output_path)
    print(f"✔ Original  : {orig_w}x{orig_h}")
    print(f"✔ Redimensionné : {new_w}x{new_h}  (ratio {ratio:.4f})")
    print(f"✔ Canvas final  : {TARGET_W}x{TARGET_H}")
    print(f"✔ Sauvegardé    : {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python resize_image.py <image> [sortie]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    resize_with_padding(input_path, output_path)
