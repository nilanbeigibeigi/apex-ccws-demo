"""
APEX · Poster QR-Code Generator
================================
Generates a high-resolution QR code PNG suitable for printing on a poster.

Usage:
    1. Install: pip install qrcode[pil] pillow
    2. Edit DEMO_URL below to your deployed Streamlit URL.
    3. Run: python generate_qr.py
    4. Output: apex_demo_qr.png (1200x1200 PNG)

Then:
    • Open APEX_Poster_v3_Landscape.html (or _Portrait.html) in your browser.
    • Open DevTools (F12) → find the .qr-square (landscape) or .qr-square-port (portrait).
    • Replace the placeholder <svg> inside it with: <img src="apex_demo_qr.png"
      style="width:100%;height:100%;object-fit:contain;border-radius:8px;">
    • Or: print the poster to PDF, then overlay the QR PNG in Preview/Acrobat.

For maximum scan reliability: target 4cm x 4cm minimum on the printed poster.
"""

import qrcode
from qrcode.constants import ERROR_CORRECT_H

# ─────── EDIT THIS ───────
DEMO_URL = "https://apex-ccws-demo.streamlit.app"   # Replace with your live demo URL
OUTPUT   = "apex_demo_qr.png"
SIZE_PX  = 1200  # Target output resolution
# ─────────────────────────


def main():
    qr = qrcode.QRCode(
        version=None,                    # Auto-size to fit URL
        error_correction=ERROR_CORRECT_H, # 30% error correction — robust when printed
        box_size=40,                     # Larger = higher resolution
        border=3,                        # 3 modules of white border (ISO standard ≥ 4, but 3 is fine here)
    )
    qr.add_data(DEMO_URL)
    qr.make(fit=True)

    img = qr.make_image(
        fill_color="#1C3A2E",           # Matches APEX forest green — stays on-brand
        back_color="#FFFFFF",
    )

    # Scale to exact target size without losing quality
    from PIL import Image
    if img.size[0] != SIZE_PX:
        img = img.resize((SIZE_PX, SIZE_PX), Image.NEAREST)

    img.save(OUTPUT, "PNG", optimize=True)
    print(f"✓ QR code saved: {OUTPUT}")
    print(f"  Resolution: {SIZE_PX}x{SIZE_PX}px")
    print(f"  URL encoded: {DEMO_URL}")
    print(f"  Print size target: 4cm x 4cm minimum for reliable scanning")


if __name__ == "__main__":
    main()
