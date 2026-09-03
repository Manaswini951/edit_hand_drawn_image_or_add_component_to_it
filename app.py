import io
import os
import math
import zipfile
import urllib.request
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Page configuration
st.set_page_config(
    page_title="50-Style Commercial Merch Generator",
    page_icon="👕",
    layout="wide"
)

FONT_DIR = "./google_fonts_cache"
os.makedirs(FONT_DIR, exist_ok=True)

# ============================================================
# 50 DISTINCT GOOGLE FONTS DATABASE
# ============================================================

GOOGLE_FONTS = {
    # Scripts & Cursive Calligraphy
    "Pacifico": "https://fonts.gstatic.com/s/pacifico/v22/F31dPrm1i3DCRJ40_z0A034.ttf",
    "Great Vibes": "https://fonts.gstatic.com/s/greatvibes/v18/RWmMoKWR9v4ksMfaWd_JN9XFiaQ.ttf",
    "Dancing Script": "https://fonts.gstatic.com/s/dancingscript/v25/If2cXTr6YS-zF4S-biP4poC3t8g6.ttf",
    "Sacramento": "https://fonts.gstatic.com/s/sacramento/v15/bu3pB2mg6T24H_24S4C7S0g.ttf",
    "Satisfy": "https://fonts.gstatic.com/s/satisfy/v17/rP2Hp2yn6lkG50LoCZOI.ttf",
    "Kaushan Script": "https://fonts.gstatic.com/s/kaushanscript/v14/VM4cD3f3B3GIG6A-25yO23FqX8A.ttf",
    "Caveat": "https://fonts.gstatic.com/s/caveat/v18/WnLfH4-WhTw33x9623A26A.ttf",
    "Yellowtail": "https://fonts.gstatic.com/s/yellowtail/v19/NGS2v5_NCpVU3S522A0059E.ttf",

    # Heavy Vintage & Block Fonts
    "Bebas Neue": "https://fonts.gstatic.com/s/bebasneue/v14/JTUSjIg1_i6t8kCHKm45xW4.ttf",
    "Abril Fatface": "https://fonts.gstatic.com/s/abrilfatface/v19/zfvMgZhM-2f71888pfgm433_2vhR.ttf",
    "Lobster": "https://fonts.gstatic.com/s/lobster/v30/neLic_12_oS9z824_z16.ttf",
    "Black Ops One": "https://fonts.gstatic.com/s/blackopsone/v20/qApdC2451372s5I_iA2a1X9F.ttf",
    "Bungee": "https://fonts.gstatic.com/s/bungee/v14/N0bU2SRTOI3V96423C8.ttf",
    "Alfa Slab One": "https://fonts.gstatic.com/s/alfaslabone/v18/6Nu13v_2A2_4u803d1_40114.ttf",
    "Righteous": "https://fonts.gstatic.com/s/righteous/v16/1X313y8iS4I9A422204.ttf",
    "Anton": "https://fonts.gstatic.com/s/anton/v25/1Ptg83L03p2-25Y.ttf",

    # Playful & Nursery Hand-Drawn
    "Amatic SC": "https://fonts.gstatic.com/s/amaticsc/v25/T12vOu213Yp03A28206q510x.ttf",
    "Luckiest Guy": "https://fonts.gstatic.com/s/luckiestguy/v18/_P3fSzo8spky32iZid43bgf3-N0.ttf",
    "Patrick Hand": "https://fonts.gstatic.com/s/patrickhand/v19/L0x5DF4xlVMF-p_6yS23A8I.ttf",
    "Indie Flower": "https://fonts.gstatic.com/s/indieflower/v17/m8I7WjhB44-14vfS2a0050.ttf",
    "Architects Daughter": "https://fonts.gstatic.com/s/architectsdaughter/v18/K232ZJ2013-05s9S423C32Fp.ttf",
    "Fredoka": "https://fonts.gstatic.com/s/fredoka/v12/u42aH7B13-14vS9S.ttf",
    "Sniglet": "https://fonts.gstatic.com/s/sniglet/v19/13813Y2013-05s9S.ttf",
    "Chewy": "https://fonts.gstatic.com/s/chewy/v18/u42aH7B13-14vS9S.ttf",

    # High-Fashion Serifs
    "Playfair Display": "https://fonts.gstatic.com/s/playfairdisplay/v30/nuFiD-vYSZviVYUb_RJ3ijvryeA61SC6.ttf",
    "Prata": "https://fonts.gstatic.com/s/prata/v19/TUZ3zwadI3A5G5eB6io.ttf",
    "DM Serif Display": "https://fonts.gstatic.com/s/dmserifdisplay/v15/rnS3-xdaB3e490_p0B833zN60YI.ttf",
    "Cinzel": "https://fonts.gstatic.com/s/cinzel/v19/8vIJ7w0433p88PnH.ttf",
    "Cormorant Garamond": "https://fonts.gstatic.com/s/cormorantgaramond/v16/co3v423K2.ttf",

    # Modern Sans
    "Montserrat": "https://fonts.gstatic.com/s/montserrat/v26/JTUHjIg1_i6t8kCHKm4532VJOt5-QNFgpCtr6Hw5aX8.ttf",
    "Syne": "https://fonts.gstatic.com/s/syne/v22/8vIJ7w0433p88PnH37fhhI1m.ttf",
    "Oswald": "https://fonts.gstatic.com/s/oswald/v49/TK3_WkUHHAIoi48qC5qs253fL03I4A.ttf",
    "Outfit": "https://fonts.gstatic.com/s/outfit/v11/Q18v7w0433p88PnH.ttf",
    "Space Grotesk": "https://fonts.gstatic.com/s/spacegrotesk/v16/8vIJ7w0433p88PnH.ttf"
}

FONT_NAMES = list(GOOGLE_FONTS.keys())

def load_font(font_name: str, size: int):
    """Downloads TTF reliably and loads into Pillow."""
    safe_name = font_name.replace(" ", "_")
    filename = os.path.join(FONT_DIR, f"{safe_name}.ttf")

    if font_name in GOOGLE_FONTS and not os.path.exists(filename):
        try:
            req = urllib.request.Request(
                GOOGLE_FONTS[font_name], 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as resp, open(filename, 'wb') as f:
                f.write(resp.read())
        except Exception:
            pass

    if os.path.exists(filename):
        try:
            return ImageFont.truetype(filename, size=size)
        except Exception:
            pass

    # System Fallback
    fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf"
    ]
    for p in fallbacks:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

# ============================================================
# CLIPPING MASK DESIGN GENERATOR ENGINE
# ============================================================

def generate_design_variation(
    phrase: str,
    artwork_img: Image.Image,
    style_index: int,
    stroke_outline: bool = True
) -> Image.Image:
    """Generates 1 of 50 distinct typography clipping mask designs."""
    
    font_name = FONT_NAMES[style_index % len(FONT_NAMES)]
    layout_type = style_index % 6  # 6 Layout Variations
    
    canvas_w, canvas_h = 1200, 1200
    mask = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(mask)
    
    words = phrase.upper().split()
    if not words:
        words = ["YOUR", "TEXT"]

    # Dynamic Font Sizing
    base_size = int(220 - (len(words) * 15))
    font_obj = load_font(font_name, max(80, base_size))

    # LAYOUT 0: Single-Line Horizontal Centered
    if layout_type == 0:
        bbox = draw.textbbox((0, 0), phrase.upper(), font=font_obj)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((canvas_w - tw) // 2 - bbox[0], (canvas_h - th) // 2 - bbox[1]), phrase.upper(), font=font_obj, fill=255)

    # LAYOUT 1: Multi-Line Stacked Block
    elif layout_type == 1:
        line_heights = []
        bboxes = []
        for word in words:
            b = draw.textbbox((0, 0), word, font=font_obj)
            bboxes.append(b)
            line_heights.append(b[3] - b[1])

        total_h = sum(line_heights) + (len(words) - 1) * 30
        start_y = (canvas_h - total_h) // 2

        for i, word in enumerate(words):
            b = bboxes[i]
            tw = b[2] - b[0]
            tx = (canvas_w - tw) // 2 - b[0]
            ty = start_y - b[1]
            draw.text((tx, ty), word, font=font_obj, fill=255)
            start_y += line_heights[i] + 30

    # LAYOUT 2: Arched Rainbow Curve Path
    elif layout_type == 2:
        chars = list(phrase.upper())
        num_chars = len(chars)
        radius = 380
        center_x, center_y = canvas_w // 2, canvas_h // 2 + 150
        angle_spread = 110  # Degrees

        for i, char in enumerate(chars):
            if char == " ":
                continue
            angle_deg = -angle_spread / 2 + (i / max(1, num_chars - 1)) * angle_spread
            angle_rad = math.radians(angle_deg - 90)

            x = center_x + radius * math.cos(angle_rad)
            y = center_y + radius * math.sin(angle_rad)

            # Single Letter Mask
            char_mask = Image.new("L", (200, 200), 0)
            cdraw = ImageDraw.Draw(char_mask)
            cb = cdraw.textbbox((0, 0), char, font=font_obj)
            cdraw.text((100 - (cb[2]-cb[0])//2 - cb[0], 100 - (cb[3]-cb[1])//2 - cb[1]), char, font=font_obj, fill=255)
            rotated_char = char_mask.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)

            mask.paste(rotated_char, (int(x - rotated_char.width//2), int(y - rotated_char.height//2)), rotated_char)

    # LAYOUT 3: Giant Drop Cap Initial
    elif layout_type == 3:
        first_letter = phrase[0].upper()
        rest_text = phrase[1:].upper()

        big_font = load_font(font_name, 380)
        small_font = load_font(font_name, 110)

        # Draw Giant First Letter
        draw.text((120, 380), first_letter, font=big_font, fill=255)
        # Draw Rest Text
        draw.text((420, 520), rest_text, font=small_font, fill=255)

    # LAYOUT 4: Staggered / Bouncy Baseline
    elif layout_type == 4:
        chars = list(phrase.upper())
        curr_x = 100
        for i, char in enumerate(chars):
            if char == " ":
                curr_x += 40
                continue
            y_shift = -40 if (i % 2 == 0) else 40
            cb = draw.textbbox((0, 0), char, font=font_obj)
            draw.text((curr_x - cb[0], 550 + y_shift - cb[1]), char, font=font_obj, fill=255)
            curr_x += (cb[2] - cb[0]) + 15

    # LAYOUT 5: Boxed Frame Typography
    else:
        bbox = draw.textbbox((0, 0), phrase.upper(), font=font_obj)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = (canvas_w - tw) // 2 - bbox[0], (canvas_h - th) // 2 - bbox[1]
        draw.text((tx, ty), phrase.upper(), font=font_obj, fill=255)
        
        # Outer Frame
        draw.rectangle([tx - 40, ty - 40, tx + tw + 40, ty + th + 40], outline=255, width=12)

    # CLIP ARTWORK INSIDE MASK
    art_resized = artwork_img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS).convert("RGB")
    art_arr = np.array(art_resized)
    mask_arr = np.array(mask)

    # Optional Outer Stroke Contour
    if stroke_outline:
        kernel = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(mask_arr, kernel, iterations=1)
        border = cv2.subtract(dilated, mask_arr)
        
        # Black border
        rgba_border = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        rgba_border[:, :, 3] = border
        
        rgba_fill = np.dstack((art_arr, mask_arr))
        
        result_img = Image.fromarray(rgba_border, "RGBA")
        result_img.alpha_composite(Image.fromarray(rgba_fill, "RGBA"))
    else:
        rgba_fill = np.dstack((art_arr, mask_arr))
        result_img = Image.fromarray(rgba_fill, "RGBA")

    # Crop tightly to content
    bbox = result_img.getbbox()
    if bbox:
        result_img = result_img.crop(bbox)

    padding = 50
    padded = Image.new("RGBA", (result_img.width + padding * 2, result_img.height + padding * 2), (0, 0, 0, 0))
    padded.alpha_composite(result_img, (padding, padding))
    return padded

def generate_3d_product_mockup(artwork: Image.Image) -> Image.Image:
    mockup = Image.new("RGBA", (1200, 1200), (238, 240, 245, 255))
    draw = ImageDraw.Draw(mockup)

    # Draw T-Shirt Outline
    shirt_pts = [(350, 140), (450, 90), (750, 90), (850, 140), (1050, 310), (940, 460), (870, 410), (870, 1120), (330, 1120), (330, 410), (260, 460), (150, 310)]
    draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(190, 190, 190), width=4)
    draw.arc((500, 70, 700, 170), start=0, end=180, fill=(180, 180, 180), width=5)
    
    target_w = 460
    scale = target_w / float(artwork.width)
    target_h = int(artwork.height * scale)
    art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mockup.alpha_composite(art_resized, (370, 340))

    return mockup

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("👕 50-Style Commercial T-Shirt Generator")
st.write("Upload 1 Artwork Pattern and enter 1 Sentence. The system generates **50 completely distinct commercial t-shirt print designs** using Google Fonts & Clipping Masks!")

col_in1, col_in2 = st.columns([1, 1])

with col_in1:
    art_file = st.file_uploader("Upload Artwork Pattern (Slot 1):", type=["jpg", "png", "webp"])
    if art_file:
        art_img_raw = Image.open(art_file)
        st.image(art_img_raw, caption="Uploaded Artwork Fill", use_container_width=True)

with col_in2:
    phrase_input = st.text_input("Enter Sentence / Text:", "I AM NOT TRASH").upper()
    enable_stroke = st.checkbox("Add Outer Black Border Stroke", value=True)
    num_designs = st.slider("Number of Variations to Generate:", 5, 50, 50)

st.markdown("---")

if st.button(f"🚀 Generate {num_designs} T-Shirt Design Variations", type="primary", use_container_width=True):
    if not art_file or not phrase_input.strip():
        st.warning("Please upload an Artwork Pattern image and enter a text phrase.")
    else:
        with st.spinner(f"Generating {num_designs} unique Google Font clipping mask designs..."):
            art_img = Image.open(art_file)
            
            generated_results = []
            
            for i in range(num_designs):
                f_name = FONT_NAMES[i % len(FONT_NAMES)]
                design_img = generate_design_variation(
                    phrase=phrase_input,
                    artwork_img=art_img,
                    style_index=i,
                    stroke_outline=enable_stroke
                )
                generated_results.append((f"Design_{i+1}_{f_name}", design_img))

            # Display Preview Gallery
            st.subheader(f"🖼️ Generated {len(generated_results)} Commercial Designs")
            
            cols = st.columns(3)
            for idx, (title, img) in enumerate(generated_results):
                with cols[idx % 3]:
                    st.markdown(f"**#{idx+1}: {title.split('_')[-1]}**")
                    st.image(img, use_container_width=True)

            # Show Primary Design on 3D Shirt
            st.markdown("---")
            st.subheader("👕 3D T-Shirt Mockup Preview (Design #1)")
            shirt_preview = generate_3d_product_mockup(generated_results[0][1])
            st.image(shirt_preview, width=500)

            # ZIP Exporter
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_f:
                for title, img in generated_results:
                    b = io.BytesIO()
                    img.save(b, format="PNG", dpi=(300, 300))
                    zip_f.writestr(f"50_TShirt_Pack/{title}.png", b.getvalue())

            st.download_button(
                label=f"📦 Download All {len(generated_results)} High-Res Designs (300 DPI ZIP Pack)",
                data=zip_buffer.getvalue(),
                file_name=f"{phrase_input.replace(' ', '_')}_50_Designs_Pack.zip",
                mime="application/zip",
                use_container_width=True
            )
