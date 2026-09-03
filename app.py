import io
import os
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

st.set_page_config(
    page_title="Drawing-to-Calligraphy Generator",
    page_icon="✍️",
    layout="wide"
)

MAX_SIZE = 1800

# ============================================================
# FONT HELPER WITH SYSTEM SEARCH & LIVE PREVIEW DATA
# ============================================================

def get_available_fonts():
    """Scans common system paths for heavy, bold, display fonts."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/coopbl.ttf",
        "C:/Windows/Fonts/arialbd.ttf"
    ]
    
    valid_fonts = {}
    for path in font_paths:
        if os.path.exists(path):
            name = os.path.basename(path).split(".")[0].upper().replace("-", " ")
            valid_fonts[f"Extra Bold - {name}"] = path
            
    if not valid_fonts:
        valid_fonts["Default System Bold"] = None
        
    return valid_fonts

AVAILABLE_FONTS = get_available_fonts()

def load_custom_font(font_path, size):
    """Loads font at specified size with fallback."""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

# ============================================================
# BACKGROUND REMOVAL & ISOLATION
# ============================================================

def resize_image(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img.copy()
    scale = max_size / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

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

    seeds = cv2.bitwise_or(relative_dark_mask, blackhat_mask)
    seeds = cv2.bitwise_or(seeds, color_mask)
    seeds = cv2.bitwise_or(seeds, edge_mask)

    very_dark = np.where(gray < 85, 255, 0).astype(np.uint8)
    structure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    nearby_structure = cv2.dilate(edge_mask, structure_kernel, iterations=1)
    protected_dark = cv2.bitwise_and(very_dark, nearby_structure)
    seeds = cv2.bitwise_or(seeds, protected_dark)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return mask

def create_transparent_drawing(img):
    original_rgb = img.convert("RGB")
    mask = extract_clean_drawing_mask(original_rgb)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask[mask < 18] = 0

    orig_arr = np.array(original_rgb)
    rgba_arr = np.dstack((orig_arr, mask))
    result = Image.fromarray(rgba_arr, "RGBA")

    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)

    padding = 20
    padded = Image.new("RGBA", (result.width + padding * 2, result.height + padding * 2), (0, 0, 0, 0))
    padded.alpha_composite(result, (padding, padding))
    return padded

# ============================================================
# ADVANCED TYPOGRAPHY COMPOSITOR
# ============================================================

def process_image_portion(source_img, crop_pct_x, crop_pct_y, zoom_level):
    """Crops and scales specific region of drawing for filling a letter."""
    w, h = source_img.size
    crop_w = int(w / zoom_level)
    crop_h = int(h / zoom_level)
    
    start_x = int((w - crop_w) * (crop_pct_x / 100.0))
    start_y = int((h - crop_h) * (crop_pct_y / 100.0))
    
    cropped = source_img.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
    return cropped

def render_thick_stroke_text_mask(letter, font, stroke_expand=0):
    """Generates letter alpha mask with artificial stroke thickness expansion."""
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), letter, font=font)
    
    lw = max(10, bbox[2] - bbox[0] + 40 + (stroke_expand * 2))
    lh = max(10, bbox[3] - bbox[1] + 40 + (stroke_expand * 2))
    
    mask = Image.new("L", (lw, lh), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # Text positioning
    tx = 20 + stroke_expand - bbox[0]
    ty = 20 + stroke_expand - bbox[1]
    
    mask_draw.text((tx, ty), letter, font=font, fill=255)
    
    # Expand thickness via dilation
    if stroke_expand > 0:
        mask_np = np.array(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_expand * 2 + 1, stroke_expand * 2 + 1))
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        mask = Image.fromarray(mask_np)
        
    return mask, lw, lh

# ============================================================
# STREAMLIT UI & WORKFLOW
# ============================================================

if "drawings" not in st.session_state:
    st.session_state["drawings"] = []

st.title("✍️ Hand-Drawn Drawing to Calligraphy Studio")
st.write("Convert sketches, doodles, and paintings into high-impact typography with per-letter texture mapping, thick bold fonts, and custom artwork alignment.")

# --- STEP 1: UPLOAD & ISOLATE DRAWINGS ---
st.header("Step 1: Upload Your Hand-Drawn Artwork")

uploaded_files = st.file_uploader(
    "Upload Hand-Drawn Photos / Sketches (JPG, PNG, WEBP):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if st.button("✂️ Extract & Prepare Artwork", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one image first.")
    else:
        extracted = []
        with st.spinner("Isolating drawing strokes from backgrounds..."):
            for idx, f in enumerate(uploaded_files):
                raw = Image.open(f)
                raw = ImageOps.exif_transpose(raw).convert("RGB")
                resized = resize_image(raw, MAX_SIZE)
                isolated = create_transparent_drawing(resized)
                extracted.append({"id": idx, "name": f.name, "image": isolated})
        st.session_state["drawings"] = extracted
        st.success(f"Isolated {len(extracted)} drawing(s) successfully!")

# Show extracted drawing previews
if st.session_state["drawings"]:
    cols = st.columns(min(4, len(st.session_state["drawings"])))
    for idx, item in enumerate(st.session_state["drawings"]):
        with cols[idx % len(cols)]:
            st.markdown(f"**Drawing #{idx + 1}**")
            st.image(item["image"], use_container_width=True)

    # --- STEP 2: TYPOGRAPHY & FONT SELECTION ---
    st.markdown("---")
    st.header("Step 2: Phrase & Heavy Display Font Styling")

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        phrase = st.text_input("Enter Phrase / Text:", "ARTWORK").upper()
    with col_t2:
        selected_font_label = st.selectbox("Select Heavy Font Style:", list(AVAILABLE_FONTS.keys()))
        selected_font_path = AVAILABLE_FONTS[selected_font_label]

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        font_size = st.slider("Font Base Size (px):", 120, 500, 220)
    with col_s2:
        letter_extra_thickness = st.slider("Add Extra Thickness (Stroke Expand):", 0, 30, 8)
    with col_s3:
        letter_spacing = st.slider("Letter Spacing (Gap):", -20, 100, 15)

    # Load Font
    active_font = load_custom_font(selected_font_path, font_size)

    # Font Style Preview Bar
    preview_img = Image.new("RGBA", (800, 100), (255, 255, 255, 255))
    p_draw = ImageDraw.Draw(preview_img)
    p_draw.text((20, 10), phrase if phrase else "PREVIEW", font=active_font, fill=(20, 20, 20))
    st.image(preview_img, caption="Font Preview", use_container_width=True)

    # --- STEP 3: MAPPING MODE & PER-LETTER FINE-TUNING ---
    st.markdown("---")
    st.header("Step 3: Map Drawings onto Letters")

    mapping_mode = st.radio(
        "Mapping Mode:",
        ["Entire Sentence (Single continuous drawing spans across whole text)", 
         "Per-Letter Assignment (Choose specific drawing & crop area for EACH letter)"],
        horizontal=True
    )

    clean_phrase = [c for c in phrase if c.strip()]
    letter_configs = {}

    if mapping_mode.startswith("Per-Letter") and clean_phrase:
        st.subheader("Fine-Tune Individual Letters")
        
        for idx, char in enumerate(clean_phrase):
            with st.expander(f"Letter #{idx + 1}: '{char}' Configuration", expanded=(idx == 0)):
                lc1, lc2, lc3, lc4 = st.columns(4)
                with lc1:
                    assigned_drawing_idx = st.selectbox(
                        f"Drawing for '{char}':",
                        options=range(len(st.session_state["drawings"])),
                        format_func=lambda x: f"Drawing #{x + 1}",
                        key=f"char_draw_{idx}"
                    )
                with lc2:
                    crop_x = st.slider(f"Horizontal Shift % ('{char}'):", 0, 100, 50, key=f"cx_{idx}")
                with lc3:
                    crop_y = st.slider(f"Vertical Shift % ('{char}'):", 0, 100, 50, key=f"cy_{idx}")
                with lc4:
                    zoom = st.slider(f"Zoom In Level ('{char}'):", 1.0, 4.0, 1.2, key=f"zoom_{idx}")

                letter_configs[idx] = {
                    "drawing_idx": assigned_drawing_idx,
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                    "zoom": zoom
                }
    else:
        # Sentence-wide configuration
        st.subheader("Global Sentence Texture Mapping")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            global_drawing_idx = st.selectbox(
                "Drawing to span across whole phrase:",
                options=range(len(st.session_state["drawings"])),
                format_func=lambda x: f"Drawing #{x + 1}"
            )
        with sc2:
            global_crop_x = st.slider("Horizontal Texture Shift %:", 0, 100, 50)
        with sc3:
            global_crop_y = st.slider("Vertical Texture Shift %:", 0, 100, 50)

    # Outline / Stroke Effects
    st.subheader("Outline & Background Settings")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        enable_outline = st.checkbox("Add Outer Contour Stroke", value=True)
        outline_color = st.color_picker("Contour Color:", "#000000")
    with bc2:
        outline_width = st.slider("Contour Width:", 1, 25, 6) if enable_outline else 0
    with bc3:
        bg_color = st.color_picker("Background Color:", "#FFFFFF")

    # --- STEP 4: COMPOSITE CALLIGRAPHY ---
    st.markdown("---")
    if st.button("🚀 Render High-Res Calligraphy Artwork", type="primary", use_container_width=True):
        if not phrase.strip():
            st.warning("Please type a phrase first.")
        else:
            with st.spinner("Rendering custom calligraphy typography..."):
                rendered_letters = []
                total_width = 0
                max_height = 0

                # Render each letter individually
                for idx, char in enumerate(phrase):
                    if char == " ":
                        space_w = int(font_size * 0.4)
                        rendered_letters.append({"is_space": True, "width": space_w})
                        total_width += space_w + letter_spacing
                        continue

                    # Generate letter mask
                    mask, lw, lh = render_thick_stroke_text_mask(
                        char, active_font, stroke_expand=letter_extra_thickness
                    )
                    
                    # Determine source drawing
                    if mapping_mode.startswith("Per-Letter"):
                        cfg = letter_configs.get(idx, {"drawing_idx": 0, "crop_x": 50, "crop_y": 50, "zoom": 1.0})
                        src_img = st.session_state["drawings"][cfg["drawing_idx"]]["image"]
                        texture_portion = process_image_portion(src_img, cfg["crop_x"], cfg["crop_y"], cfg["zoom"])
                    else:
                        src_img = st.session_state["drawings"][global_drawing_idx]["image"]
                        texture_portion = process_image_portion(src_img, global_crop_x, global_crop_y, 1.0)

                    # Scale portion to letter mask size
                    texture_resized = texture_portion.resize((lw, lh), Image.Resampling.LANCZOS).convert("RGBA")
                    
                    # Apply text mask
                    letter_tile = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
                    letter_tile.paste(texture_resized, (0, 0), mask)

                    # Optional Contour Outline
                    if enable_outline and outline_width > 0:
                        outline_mask = np.array(mask)
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_width * 2 + 1, outline_width * 2 + 1))
                        dilated = cv2.dilate(outline_mask, k, iterations=1)
                        border_mask = cv2.subtract(dilated, outline_mask)
                        
                        rgb_border = tuple(int(outline_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        stroke_img = Image.new("RGBA", (lw, lh), rgb_border + (255,))
                        
                        combined_letter = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
                        combined_letter.paste(stroke_img, (0, 0), Image.fromarray(border_mask))
                        combined_letter.alpha_composite(letter_tile, (0, 0))
                        letter_tile = combined_letter

                    rendered_letters.append({
                        "is_space": False,
                        "image": letter_tile,
                        "width": lw,
                        "height": lh
                    })

                    total_width += lw + letter_spacing
                    max_height = max(max_height, lh)

                # Assemble onto final canvas
                canvas_padding = 100
                canvas_w = total_width + (canvas_padding * 2)
                canvas_h = max_height + (canvas_padding * 2)

                bg_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                final_canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_rgb + (255,))

                curr_x = canvas_padding
                for item in rendered_letters:
                    if item["is_space"]:
                        curr_x += item["width"] + letter_spacing
                    else:
                        curr_y = canvas_padding + (max_height - item["height"]) // 2
                        final_canvas.alpha_composite(item["image"], (curr_x, curr_y))
                        curr_x += item["width"] + letter_spacing

                st.subheader("🖼️ Generated Calligraphy Result")
                st.image(final_canvas.convert("RGB"), use_container_width=True)

                # Download File
                buf = io.BytesIO()
                final_canvas.convert("RGB").save(buf, format="PNG")
                st.download_button(
                    label="📥 Download High-Res Calligraphy (PNG)",
                    data=buf.getvalue(),
                    file_name="handdrawn_calligraphy_artwork.png",
                    mime="image/png",
                    use_container_width=True
                )
