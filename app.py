import io
import os
import math
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter,
    ImageOps
)

# Page configuration
st.set_page_config(
    page_title="Commercial Typographic Graphic Studio",
    page_icon="🎨",
    layout="wide"
)

MAX_SIZE = 1800
AUTO_STRAIGHTEN_DRAWING = True
MAX_STRAIGHTEN_ANGLE = 18

# Local font directory
FONT_DIR = "./assets/fonts"
os.makedirs(FONT_DIR, exist_ok=True)

# ============================================================
# LOCAL FONT MANAGEMENT & RELIABLE LOADING
# ============================================================

# Define font styles mapped to local files or OS paths
FONT_CATALOG = {
    "Cursive Script - Elegant": [
        os.path.join(FONT_DIR, "script.ttf"),
        "/usr/share/fonts/truetype/freefont/FreeScript.ttf",
        "C:/Windows/Fonts/BRUSHSCI.TTF"
    ],
    "Bold Display - Impact": [
        os.path.join(FONT_DIR, "display.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/impact.ttf"
    ],
    "Serif - Boutique": [
        os.path.join(FONT_DIR, "serif.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "C:/Windows/Fonts/georgiab.ttf"
    ],
    "Clean Sans - Minimal": [
        os.path.join(FONT_DIR, "sans.ttf"),
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf"
    ]
}

def load_reliable_font(style_key, size):
    """Loads font from local paths; falls back safely to system TrueType fonts."""
    paths = FONT_CATALOG.get(style_key, [])
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    
    # Fallback to standard system TTFs
    system_fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf"
    ]
    for p in system_fallbacks:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
                
    return ImageFont.load_default()

# ============================================================
# DECORATIVE GRAPHIC ACCENTS GENERATOR (Banners, Flourishes, Stars)
# ============================================================

def draw_ribbon_banner(width, height, fill_color=(0, 0, 0, 255)):
    """Generates a curved ribbon banner asset."""
    banner = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(banner)
    
    pad = 10
    w, h = width - pad * 2, height - pad * 2
    
    points = [
        (pad, pad + 15),
        (pad + w * 0.5, pad),
        (pad + w, pad + 15),
        (pad + w - 15, pad + h - 10),
        (pad + w * 0.5, pad + h),
        (pad + 15, pad + h - 10)
    ]
    draw.polygon(points, fill=fill_color)
    return banner

def draw_decorative_swash(width, height, fill_color=(0, 0, 0, 255)):
    """Generates decorative underline flourish loops."""
    swash = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(swash)
    
    mid_y = height // 2
    draw.arc([10, 5, width // 2, height - 5], start=0, end=180, fill=fill_color, width=6)
    draw.arc([width // 2 - 10, 5, width - 10, height - 5], start=180, end=360, fill=fill_color, width=6)
    return swash

def draw_accent_star(size, fill_color=(0, 0, 0, 255)):
    """Draws a 5-point accent star."""
    star = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(star)
    
    cx, cy = size / 2, size / 2
    r_outer = size / 2 - 2
    r_inner = r_outer * 0.4
    points = []
    
    for i in range(10):
        r = r_outer if i % 2 == 0 else r_inner
        angle = i * math.pi / 5 - math.pi / 2
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        
    draw.polygon(points, fill=fill_color)
    return star

# ============================================================
# EXACT SHADOW-IMMUNE EXTRACTION ENGINE
# ============================================================

def resize_image(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img.copy()
    scale = max_size / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def remove_small_components(mask, min_area=18):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned

def extract_clean_drawing_mask(img_rgb):
    arr = np.array(img_rgb).astype(np.uint8)
    h, w = arr.shape[:2]

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    bg_size = max(61, int(min(h, w) * 0.10))
    if bg_size % 2 == 0:
        bg_size += 1

    bg_gray = cv2.GaussianBlur(gray, (bg_size, bg_size), 0)
    local_darkness = bg_gray.astype(np.float32) - gray.astype(np.float32)

    sat_blur = cv2.GaussianBlur(sat, (35, 35), 0)
    color_difference = np.abs(sat - sat_blur)
    color_mask = np.where((sat > 38) | (color_difference > 14), 255, 0).astype(np.uint8)

    bh_size = max(21, int(min(h, w) * 0.035))
    if bh_size % 2 == 0:
        bh_size += 1

    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_size, bh_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    blackhat_mask = np.where(blackhat > 10, 255, 0).astype(np.uint8)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_mask = np.where(magnitude > 24.0, 255, 0).astype(np.uint8)

    relative_dark_mask = np.where(local_darkness > 13.0, 255, 0).astype(np.uint8)
    strong_relative_dark = np.where(local_darkness > 24.0, 255, 0).astype(np.uint8)

    seeds = cv2.bitwise_or(relative_dark_mask, blackhat_mask)
    seeds = cv2.bitwise_or(seeds, color_mask)
    seeds = cv2.bitwise_or(seeds, edge_mask)

    very_dark = np.where(gray < 85, 255, 0).astype(np.uint8)
    structure_kernel_size = max(9, int(min(h, w) * 0.012))
    if structure_kernel_size % 2 == 0:
        structure_kernel_size += 1

    structure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_kernel_size, structure_kernel_size))
    nearby_structure = cv2.dilate(edge_mask, structure_kernel, iterations=1)
    protected_dark = cv2.bitwise_and(very_dark, nearby_structure)
    seeds = cv2.bitwise_or(seeds, protected_dark)

    close_size = max(5, int(min(h, w) * 0.008))
    if close_size % 2 == 0:
        close_size += 1

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    local_mean = cv2.GaussianBlur(gray, (31, 31), 0)
    local_difference = cv2.absdiff(gray, local_mean)
    texture_signal = np.where(local_difference > 7, 255, 0).astype(np.uint8)

    smooth_shadow = (
        (mask > 0) &
        (texture_signal == 0) &
        (color_mask == 0) &
        (strong_relative_dark == 0)
    )
    mask[smooth_shadow] = 0

    mask = remove_small_components(mask, min_area=18)

    bw = max(6, int(min(h, w) * 0.004))
    mask[:bw, :] = 0
    mask[-bw:, :] = 0
    mask[:, :bw] = 0
    mask[:, -bw:] = 0

    return mask

def create_transparent_drawing(img):
    original_rgb = img.convert("RGB")
    mask = extract_clean_drawing_mask(original_rgb)
    mask = remove_small_components(mask, min_area=18)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask[mask < 18] = 0

    orig_arr = np.array(original_rgb)
    rgba_arr = np.dstack((orig_arr, mask))
    result = Image.fromarray(rgba_arr, "RGBA")

    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)

    padding = 40
    padded = Image.new("RGBA", (result.width + padding * 2, result.height + padding * 2), (0, 0, 0, 0))
    padded.alpha_composite(result, (padding, padding))
    return padded

# ============================================================
# MULTI-LINE GRAPHIC GRAPHIC COMPOSITOR (Apparel Badge Engine)
# ============================================================

def render_multi_line_graphic_quote(
    line1_text, line2_text, line3_text, line4_text,
    script_font_key, block_font_key,
    texture_img, styling
):
    """Composes multi-line graphic layouts with ribbons, star accents, and mixed font pairings."""
    canvas_w, canvas_h = 1000, 1200
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    font_script = load_reliable_font(script_font_key, 180)
    font_block = load_reliable_font(block_font_key, 160)
    font_sub = load_reliable_font(block_font_key, 90)
    
    y_offset = 100
    
    # LINE 1: Cursive Script + Star
    if line1_text.strip():
        txt_mask = Image.new("L", (canvas_w, 220), 0)
        d = ImageDraw.Draw(txt_mask)
        d.text((canvas_w // 2, 100), line1_text, font=font_script, fill=255, anchor="mm")
        
        line1_layer = Image.new("RGBA", (canvas_w, 220), (0, 0, 0, 0))
        tex_resized = texture_img.resize((canvas_w, 220), Image.Resampling.LANCZOS)
        line1_layer.paste(tex_resized, (0, 0), txt_mask)
        
        # Add Star Accent above line 1
        star = draw_accent_star(50, fill_color=(40, 40, 40, 255))
        canvas.alpha_composite(star, (canvas_w // 2 + 180, y_offset - 20))
        canvas.alpha_composite(line1_layer, (0, y_offset))
        y_offset += 200

    # LINE 2: Ribbon Banner Subtext
    if line2_text.strip():
        banner = draw_ribbon_banner(450, 110, fill_color=(30, 30, 30, 255))
        b_draw = ImageDraw.Draw(banner)
        b_draw.text((225, 55), line2_text, font=font_sub, fill=(255, 255, 255, 255), anchor="mm")
        
        canvas.alpha_composite(banner, ((canvas_w - 450) // 2, y_offset))
        y_offset += 140

    # LINE 3: Main Cursive Script + Swash Underline
    if line3_text.strip():
        txt_mask = Image.new("L", (canvas_w, 240), 0)
        d = ImageDraw.Draw(txt_mask)
        d.text((canvas_w // 2, 100), line3_text, font=font_script, fill=255, anchor="mm")
        
        line3_layer = Image.new("RGBA", (canvas_w, 240), (0, 0, 0, 0))
        tex_resized = texture_img.resize((canvas_w, 240), Image.Resampling.LANCZOS)
        line3_layer.paste(tex_resized, (0, 0), txt_mask)
        
        # Swash underline
        swash = draw_decorative_swash(400, 60, fill_color=(40, 40, 40, 255))
        canvas.alpha_composite(swash, ((canvas_w - 400) // 2, y_offset + 170))
        canvas.alpha_composite(line3_layer, (0, y_offset))
        y_offset += 250

    # LINE 4: Heavy Bold Display Block
    if line4_text.strip():
        txt_mask = Image.new("L", (canvas_w, 240), 0)
        d = ImageDraw.Draw(txt_mask)
        d.text((canvas_w // 2, 120), line4_text, font=font_block, fill=255, anchor="mm")
        
        line4_layer = Image.new("RGBA", (canvas_w, 240), (0, 0, 0, 0))
        tex_resized = texture_img.resize((canvas_w, 240), Image.Resampling.LANCZOS)
        line4_layer.paste(tex_resized, (0, 0), txt_mask)
        
        canvas.alpha_composite(line4_layer, (0, y_offset))

    # Apply Contour Outline & Shadow if enabled
    if styling.get("enable_outline"):
        out_w = styling.get("outline_width", 4)
        alpha = np.array(canvas.getchannel("A"))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (out_w * 2 + 1, out_w * 2 + 1))
        dilated = cv2.dilate(alpha, k, iterations=1)
        border_mask = cv2.subtract(dilated, alpha)
        
        stroke_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
        outlined_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        outlined_canvas.paste(stroke_img, (0, 0), Image.fromarray(border_mask))
        outlined_canvas.alpha_composite(canvas, (0, 0))
        canvas = outlined_canvas

    return canvas

def generate_3d_product_mockup(artwork: Image.Image, apparel_style: str) -> Image.Image:
    mockup = Image.new("RGBA", (1200, 1200), (238, 240, 245, 255))
    draw = ImageDraw.Draw(mockup)

    if apparel_style == "Men's Classic Crew Neck T-Shirt":
        shirt_pts = [(350, 140), (450, 90), (750, 90), (850, 140), (1050, 310), (940, 460), (870, 410), (870, 1120), (330, 1120), (330, 410), (260, 460), (150, 310)]
        draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(190, 190, 190), width=4)
        draw.arc((500, 70, 700, 170), start=0, end=180, fill=(180, 180, 180), width=5)
        
        target_w = 460
        scale = target_w / float(artwork.width)
        target_h = int(artwork.height * scale)
        art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (370, 320))
    else: # Tote Bag
        draw.line((400, 80, 400, 380), fill=(180, 160, 130), width=32)
        draw.line((800, 80, 800, 380), fill=(180, 160, 130), width=32)
        draw.polygon([(240, 380), (960, 380), (910, 1120), (290, 1120)], fill=(248, 244, 230), outline=(190, 180, 160), width=5)
        
        target_w = 480
        scale = min(480 / artwork.width, 480 / artwork.height)
        new_w, new_h = int(artwork.width * scale), int(artwork.height * scale)
        art_resized = artwork.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (240 + (720 - new_w) // 2, 380 + (740 - new_h) // 2))

    return mockup

# ============================================================
# STREAMLIT APPLICATION STATE & WORKFLOW
# ============================================================

if "drawings" not in st.session_state:
    st.session_state["drawings"] = []

st.title("🎨 Commercial Graphic Typographic Studio")
st.write("Build multi-line quote graphics with ribbon banners, scripts, and artwork textures.")

# --- STEP 1: UPLOAD & ISOLATE DRAWINGS ---
st.header("Step 1: Upload Your Artwork / Texture Image")

uploaded_files = st.file_uploader(
    "Upload Hand-Drawn Photos / Paintings / Textures (JPG, PNG):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if st.button("✂️ Extract Drawing Strokes", type="primary"):
    if not uploaded_files:
        st.warning("Please upload an image first.")
    else:
        extracted = []
        with st.spinner("Extracting drawing strokes..."):
            for idx, f in enumerate(uploaded_files):
                raw = Image.open(f)
                raw = ImageOps.exif_transpose(raw).convert("RGB")
                resized = resize_image(raw, MAX_SIZE)
                isolated = create_transparent_drawing(resized)
                extracted.append({"id": idx, "name": f.name, "image": isolated})
        st.session_state["drawings"] = extracted
        st.success(f"Isolated {len(extracted)} artwork texture(s)!")

if st.session_state["drawings"]:
    cols = st.columns(min(4, len(st.session_state["drawings"])))
    for idx, item in enumerate(st.session_state["drawings"]):
        with cols[idx % len(cols)]:
            st.markdown(f"**Artwork Texture #{idx + 1}**")
            st.image(item["image"], use_container_width=True)

    # --- STEP 2: MULTI-LINE TEXT ENTRY ---
    st.markdown("---")
    st.header("Step 2: Enter Multi-Line Layout Phrase & Font Pairing")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        line1 = st.text_input("Line 1 (Cursive Header):", "this")
        line2 = st.text_input("Line 2 (Ribbon Subtext):", "IS MY")
    with col_l2:
        line3 = st.text_input("Line 3 (Main Cursive Script):", "Writing")
        line4 = st.text_input("Line 4 (Heavy Block Text):", "SHIRT")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        script_font_choice = st.selectbox("Select Script Font:", list(FONT_CATALOG.keys()), index=0)
    with col_f2:
        block_font_choice = st.selectbox("Select Block Font:", list(FONT_CATALOG.keys()), index=1)

    # --- STEP 3: STYLING & MOCKUP OPTIONS ---
    st.markdown("---")
    st.header("Step 3: Styling & Merchandise Mockup")

    st1, st2, st3 = st.columns(3)
    with st1:
        enable_outline = st.checkbox("Add Contour Stroke", value=True)
        outline_width = st.slider("Stroke Width:", 1, 15, 5) if enable_outline else 0
    with st2:
        mockup_choice = st.selectbox("Select Merchandise Mockup:", ["Men's Classic Crew Neck T-Shirt", "Boutique Tote Bag"])
    with st3:
        assigned_texture_idx = st.selectbox(
            "Select Artwork Texture to Fill Design:",
            options=range(len(st.session_state["drawings"])),
            format_func=lambda x: f"Artwork Texture #{x + 1}"
        )

    styling_opts = {
        "enable_outline": enable_outline,
        "outline_width": outline_width
    }

    # --- STEP 4: COMPOSITE & EXPORT ---
    st.markdown("---")
    st.header("Step 4: Render Graphic Design & 3D Mockup")

    if st.button("🚀 Render Graphic Typography Design", type="primary", use_container_width=True):
        with st.spinner("Compositing graphic layout, ribbon banner, and artwork texture..."):
            active_texture = st.session_state["drawings"][assigned_texture_idx]["image"]
            
            final_graphic = render_multi_line_graphic_quote(
                line1_text=line1,
                line2_text=line2,
                line3_text=line3,
                line4_text=line4,
                script_font_key=script_font_choice,
                block_font_key=block_font_choice,
                texture_img=active_texture,
                styling=styling_opts
            )

            mockup_img = generate_3d_product_mockup(final_graphic, mockup_choice)

            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.subheader("🖼️ High-Res Graphic Design Output")
                st.image(final_graphic, use_container_width=True)
            with res_c2:
                st.subheader(f"👕 Live 3D {mockup_choice} Mockup")
                st.image(mockup_img, use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                buf1 = io.BytesIO()
                final_graphic.save(buf1, format="PNG", dpi=(300, 300))
                st.download_button(
                    label="📥 Download Graphic Design PNG",
                    data=buf1.getvalue(),
                    file_name="Graphic_Typography_Design.png",
                    mime="image/png",
                    use_container_width=True
                )
            with dl2:
                buf2 = io.BytesIO()
                mockup_img.save(buf2, format="PNG")
                st.download_button(
                    label="📥 Download 3D Mockup PNG",
                    data=buf2.getvalue(),
                    file_name="3D_Merch_Mockup.png",
                    mime="image/png",
                    use_container_width=True
                )
