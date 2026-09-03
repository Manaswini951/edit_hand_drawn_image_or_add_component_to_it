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
    ImageOps,
    ImageEnhance
)

# Page configuration
st.set_page_config(
    page_title="Commercial Typography & Custom Merch Studio",
    page_icon="🎨",
    layout="wide"
)

MAX_SIZE = 1800
AUTO_STRAIGHTEN_DRAWING = True
MAX_STRAIGHTEN_ANGLE = 18

# ============================================================
# FONT HELPER WITH SYSTEM SEARCH
# ============================================================

def get_available_fonts():
    """Scans common system paths for heavy display fonts."""
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

    # 1. ESTIMATE PAPER/BACKGROUND
    bg_size = max(61, int(min(h, w) * 0.10))
    if bg_size % 2 == 0:
        bg_size += 1

    bg_gray = cv2.GaussianBlur(gray, (bg_size, bg_size), 0)
    local_darkness = bg_gray.astype(np.float32) - gray.astype(np.float32)

    # 2. LOCAL COLOR DIFFERENCE
    sat_blur = cv2.GaussianBlur(sat, (35, 35), 0)
    color_difference = np.abs(sat - sat_blur)
    color_mask = np.where((sat > 38) | (color_difference > 14), 255, 0).astype(np.uint8)

    # 3. BLACK-HAT FOR STROKES
    bh_size = max(21, int(min(h, w) * 0.035))
    if bh_size % 2 == 0:
        bh_size += 1

    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_size, bh_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    blackhat_mask = np.where(blackhat > 10, 255, 0).astype(np.uint8)

    # 4. GRADIENT
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_mask = np.where(magnitude > 24.0, 255, 0).astype(np.uint8)

    # 5. RELATIVE DARKNESS
    relative_dark_mask = np.where(local_darkness > 13.0, 255, 0).astype(np.uint8)
    strong_relative_dark = np.where(local_darkness > 24.0, 255, 0).astype(np.uint8)

    # 6. SEEDS
    seeds = cv2.bitwise_or(relative_dark_mask, blackhat_mask)
    seeds = cv2.bitwise_or(seeds, color_mask)
    seeds = cv2.bitwise_or(seeds, edge_mask)

    # 7. PROTECTED DARK
    very_dark = np.where(gray < 85, 255, 0).astype(np.uint8)
    structure_kernel_size = max(9, int(min(h, w) * 0.012))
    if structure_kernel_size % 2 == 0:
        structure_kernel_size += 1

    structure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_kernel_size, structure_kernel_size))
    nearby_structure = cv2.dilate(edge_mask, structure_kernel, iterations=1)
    protected_dark = cv2.bitwise_and(very_dark, nearby_structure)
    seeds = cv2.bitwise_or(seeds, protected_dark)

    # 8. FILL REGIONS
    close_size = max(5, int(min(h, w) * 0.008))
    if close_size % 2 == 0:
        close_size += 1

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # 9. REMOVE SMOOTH SHADOWS
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

    # 10. CLEANUP
    mask = remove_small_components(mask, min_area=18)

    # 11. MARGINS
    bw = max(6, int(min(h, w) * 0.004))
    mask[:bw, :] = 0
    mask[-bw:, :] = 0
    mask[:, :bw] = 0
    mask[:, -bw:] = 0

    return mask

def straighten_drawing(image):
    if not AUTO_STRAIGHTEN_DRAWING:
        return image
    alpha = np.array(image.getchannel("A"))
    ys, xs = np.where(alpha > 50)
    if len(xs) < 100:
        return image
    points = np.column_stack((xs, ys)).astype(np.float32)
    try:
        _, eigenvectors = cv2.PCACompute(points, mean=None)
        vector = eigenvectors[0]
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        bbox_w = xs.max() - xs.min()
        bbox_h = ys.max() - ys.min()
        if bbox_h >= bbox_w:
            correction = (90 - angle) if angle > 0 else (-90 - angle)
        else:
            correction = -angle

        while correction > 90:
            correction -= 180
        while correction < -90:
            correction += 180

        if abs(correction) <= MAX_STRAIGHTEN_ANGLE:
            image = image.rotate(correction, resample=Image.Resampling.BICUBIC, expand=True)
    except Exception:
        pass
    return image

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
    result = padded

    result = straighten_drawing(result)
    return result

# ============================================================
# HIGH-END TYPOGRAPHY & MOCKUP GENERATORS
# ============================================================

def process_image_portion(source_img, crop_pct_x, crop_pct_y, zoom_level):
    w, h = source_img.size
    crop_w = int(w / zoom_level)
    crop_h = int(h / zoom_level)
    start_x = int((w - crop_w) * (crop_pct_x / 100.0))
    start_y = int((h - crop_h) * (crop_pct_y / 100.0))
    return source_img.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))

def render_letter_mask(letter, font, stroke_expand=0):
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), letter, font=font)
    
    lw = max(10, bbox[2] - bbox[0] + 40 + (stroke_expand * 2))
    lh = max(10, bbox[3] - bbox[1] + 40 + (stroke_expand * 2))
    
    mask = Image.new("L", (lw, lh), 0)
    mask_draw = ImageDraw.Draw(mask)
    tx = 20 + stroke_expand - bbox[0]
    ty = 20 + stroke_expand - bbox[1]
    
    mask_draw.text((tx, ty), letter, font=font, fill=255)
    
    if stroke_expand > 0:
        mask_np = np.array(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_expand * 2 + 1, stroke_expand * 2 + 1))
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        mask = Image.fromarray(mask_np)
        
    return mask, lw, lh

def render_typography_artwork(phrase, font, drawings, mapping_mode, letter_configs, global_cfg, styling):
    rendered_letters = []
    total_width = 0
    max_height = 0
    letter_spacing = styling.get("letter_spacing", 15)
    stroke_expand = styling.get("stroke_expand", 8)

    clean_idx = 0
    for char in phrase:
        if char == " ":
            space_w = int(font.size * 0.4)
            rendered_letters.append({"is_space": True, "width": space_w})
            total_width += space_w + letter_spacing
            continue

        mask, lw, lh = render_letter_mask(char, font, stroke_expand=stroke_expand)
        
        if mapping_mode.startswith("Per-Letter"):
            cfg = letter_configs.get(clean_idx, {"drawing_idx": 0, "crop_x": 50, "crop_y": 50, "zoom": 1.0})
            src_img = drawings[cfg["drawing_idx"]]["image"]
            texture_portion = process_image_portion(src_img, cfg["crop_x"], cfg["crop_y"], cfg["zoom"])
            clean_idx += 1
        else:
            src_img = drawings[global_cfg["drawing_idx"]]["image"]
            texture_portion = process_image_portion(src_img, global_cfg["crop_x"], global_cfg["crop_y"], 1.0)

        texture_resized = texture_portion.resize((lw, lh), Image.Resampling.LANCZOS).convert("RGBA")
        letter_tile = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        letter_tile.paste(texture_resized, (0, 0), mask)

        # Apply Drop Shadow Effect
        if styling.get("enable_shadow"):
            shadow_mask = mask.filter(ImageFilter.GaussianBlur(styling.get("shadow_blur", 10)))
            shadow_tile = Image.new("RGBA", (lw + 20, lh + 20), (0, 0, 0, 0))
            shadow_color = (0, 0, 0, 140)
            shadow_img = Image.new("RGBA", (lw, lh), shadow_color)
            shadow_tile.paste(shadow_img, (10, 10), shadow_mask)
            
            combined = Image.new("RGBA", (lw + 20, lh + 20), (0, 0, 0, 0))
            combined.alpha_composite(shadow_tile)
            combined.alpha_composite(letter_tile, (0, 0))
            letter_tile = combined
            lw, lh = lw + 20, lh + 20

        # Apply Outer Contour Stroke
        if styling.get("enable_outline") and styling.get("outline_width", 0) > 0:
            out_w = styling["outline_width"]
            outline_mask = np.array(mask)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (out_w * 2 + 1, out_w * 2 + 1))
            dilated = cv2.dilate(outline_mask, k, iterations=1)
            border_mask = cv2.subtract(dilated, outline_mask)
            
            rgb_border = tuple(int(styling["outline_color"].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            stroke_img = Image.new("RGBA", (lw, lh), rgb_border + (255,))
            
            combined = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
            combined.paste(stroke_img, (0, 0), Image.fromarray(border_mask))
            combined.alpha_composite(letter_tile, (0, 0))
            letter_tile = combined

        rendered_letters.append({"is_space": False, "image": letter_tile, "width": lw, "height": lh})
        total_width += lw + letter_spacing
        max_height = max(max_height, lh)

    canvas_padding = 80
    canvas_w = total_width + (canvas_padding * 2)
    canvas_h = max_height + (canvas_padding * 2)
    bg_color = styling.get("bg_color", "#FFFFFF")
    
    if bg_color == "TRANSPARENT":
        final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    else:
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

    return final_canvas

def generate_product_mockup(artwork: Image.Image, product_type: str) -> Image.Image:
    """Simulates real-world merchandise product mockups."""
    if product_type == "T-Shirt":
        mockup = Image.new("RGBA", (1200, 1200), (240, 240, 240, 255))
        draw = ImageDraw.Draw(mockup)
        # Draw T-shirt Silhouette
        shirt_pts = [(350, 150), (450, 100), (750, 100), (850, 150), (1050, 300), (950, 450), (880, 400), (880, 1100), (320, 1100), (320, 400), (250, 450), (150, 300)]
        draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(200, 200, 200), width=4)
        # Neck collar
        draw.arc((500, 80, 700, 180), start=0, end=180, fill=(200, 200, 200), width=4)
        
        # Fit Artwork onto Chest
        target_w = 480
        scale = target_w / float(artwork.width)
        target_h = int(artwork.height * scale)
        art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (360, 380))

    elif product_type == "Framed Canvas":
        mockup = Image.new("RGBA", (1200, 1000), (230, 225, 215, 255))
        draw = ImageDraw.Draw(mockup)
        # Wall Frame
        draw.rectangle((150, 100, 1050, 900), fill=(250, 250, 250), outline=(60, 40, 30), width=28)
        draw.rectangle((180, 130, 1020, 870), fill=(255, 255, 255), outline=(220, 220, 220), width=3)
        
        target_w = 700
        scale = min(700 / artwork.width, 600 / artwork.height)
        new_w, new_h = int(artwork.width * scale), int(artwork.height * scale)
        art_resized = artwork.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        ax = 180 + (840 - new_w) // 2
        ay = 130 + (740 - new_h) // 2
        mockup.alpha_composite(art_resized, (ax, ay))

    elif product_type == "Coffee Mug":
        mockup = Image.new("RGBA", (1000, 1000), (245, 245, 250, 255))
        draw = ImageDraw.Draw(mockup)
        # Handle
        draw.arc((650, 300, 880, 700), start=-90, end=90, fill=(220, 220, 220), width=45)
        # Mug Body
        draw.rounded_rectangle((250, 180, 700, 820), radius=30, fill=(255, 255, 255), outline=(210, 210, 210), width=6)
        
        target_w = 380
        scale = min(380 / artwork.width, 450 / artwork.height)
        new_w, new_h = int(artwork.width * scale), int(artwork.height * scale)
        art_resized = artwork.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (250 + (450 - new_w) // 2, 180 + (640 - new_h) // 2))

    else: # Tote Bag
        mockup = Image.new("RGBA", (1100, 1100), (240, 238, 230, 255))
        draw = ImageDraw.Draw(mockup)
        # Straps
        draw.line((380, 100, 380, 350), fill=(180, 160, 130), width=30)
        draw.line((720, 100, 720, 350), fill=(180, 160, 130), width=30)
        # Bag Body
        draw.polygon([(220, 350), (880, 350), (840, 1000), (260, 1000)], fill=(245, 240, 225), outline=(200, 190, 170), width=5)
        
        target_w = 450
        scale = min(450 / artwork.width, 450 / artwork.height)
        new_w, new_h = int(artwork.width * scale), int(artwork.height * scale)
        art_resized = artwork.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (220 + (620 - new_w) // 2, 350 + (650 - new_h) // 2))

    return mockup

# ============================================================
# STREAMLIT APPLICATION STATE & WORKFLOW
# ============================================================

if "drawings" not in st.session_state:
    st.session_state["drawings"] = []

st.title("🛍️ Commercial Typography & Product Merch Studio")
st.write("Convert hand drawings into commercial nursery prints, apparel typography, wedding monograms, and digital Etsy asset packs.")

# --- STEP 1: UPLOAD & ISOLATE DRAWINGS ---
st.header("Step 1: Upload Your Hand-Drawn Artwork / Photos")

uploaded_files = st.file_uploader(
    "Upload Hand-Drawn Photos / Paintings / Doodles (JPG, PNG, WEBP):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if st.button("✂️ Extract Drawings (Exact Shadow-Immune Pipeline)", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one image first.")
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
        st.success(f"Isolated {len(extracted)} drawing(s) successfully!")

if st.session_state["drawings"]:
    cols = st.columns(min(4, len(st.session_state["drawings"])))
    for idx, item in enumerate(st.session_state["drawings"]):
        with cols[idx % len(cols)]:
            st.markdown(f"**Drawing #{idx + 1}**")
            st.image(item["image"], use_container_width=True)

    # --- STEP 2: COMMERCIAL PRESET SELECTOR ---
    st.markdown("---")
    st.header("Step 2: Choose Commercial Application Preset")

    preset_use_case = st.selectbox(
        "Select Targeted Product Use Case:",
        ["Custom Nursery Name Art", "Indie Merch & T-Shirt Typography", "Wedding Monogram & Event Branding", "Etsy Commercial Alphabet Pack (A-Z Exporter)"]
    )

    if preset_use_case == "Custom Nursery Name Art":
        default_text = "LEO"
        default_font_size = 260
    elif preset_use_case == "Indie Merch & T-Shirt Typography":
        default_text = "CREATIVE"
        default_font_size = 220
    elif preset_use_case == "Wedding Monogram & Event Branding":
        default_text = "A & M"
        default_font_size = 300
    else:
        default_text = "ALPHABET"
        default_font_size = 200

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        phrase = st.text_input("Enter Text / Phrase:", default_text).upper()
    with col_t2:
        selected_font_label = st.selectbox("Select Heavy Font Style:", list(AVAILABLE_FONTS.keys()))
        selected_font_path = AVAILABLE_FONTS[selected_font_label]

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        font_size = st.slider("Font Base Size (px):", 120, 500, default_font_size)
    with col_s2:
        stroke_expand = st.slider("Extra Letter Thickness:", 0, 30, 10)
    with col_s3:
        letter_spacing = st.slider("Letter Spacing (Gap):", -20, 100, 20)

    active_font = load_custom_font(selected_font_path, font_size)

    # --- STEP 3: PER-LETTER MAPPING ---
    st.markdown("---")
    st.header("Step 3: Texture & Letter Mapping")

    mapping_mode = st.radio(
        "Mapping Mode:",
        ["Per-Letter Assignment (Choose drawing & crop region for EACH letter)",
         "Entire Phrase (Single drawing spans across whole text)"],
        horizontal=True
    )

    clean_phrase = [c for c in phrase if c.strip()]
    letter_configs = {}
    global_cfg = {"drawing_idx": 0, "crop_x": 50, "crop_y": 50}

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
                    zoom = st.slider(f"Zoom Level ('{char}'):", 1.0, 4.0, 1.2, key=f"zoom_{idx}")

                letter_configs[idx] = {
                    "drawing_idx": assigned_drawing_idx,
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                    "zoom": zoom
                }
    else:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            global_cfg["drawing_idx"] = st.selectbox(
                "Drawing to span across phrase:",
                options=range(len(st.session_state["drawings"])),
                format_func=lambda x: f"Drawing #{x + 1}"
            )
        with sc2:
            global_cfg["crop_x"] = st.slider("Horizontal Texture Shift %:", 0, 100, 50)
        with sc3:
            global_cfg["crop_y"] = st.slider("Vertical Texture Shift %:", 0, 100, 50)

    # Style Controls
    st.subheader("Professional Styling & FX")
    fx_col1, fx_col2, fx_col3, fx_col4 = st.columns(4)
    with fx_col1:
        enable_outline = st.checkbox("Add Outer Contour Stroke", value=True)
        outline_color = st.color_picker("Stroke Color:", "#000000")
        outline_width = st.slider("Stroke Width:", 1, 25, 6) if enable_outline else 0
    with fx_col2:
        enable_shadow = st.checkbox("Add Drop Shadow", value=True)
        shadow_blur = st.slider("Shadow Blur:", 2, 25, 10) if enable_shadow else 0
    with fx_col3:
        bg_transparent = st.checkbox("Transparent Canvas (PNG)", value=True)
        bg_color = "TRANSPARENT" if bg_transparent else st.color_picker("Background Color:", "#FFFFFF")
    with fx_col4:
        mockup_choice = st.selectbox("Live Product Mockup Preview:", ["T-Shirt", "Framed Canvas", "Coffee Mug", "Tote Bag"])

    styling_opts = {
        "letter_spacing": letter_spacing,
        "stroke_expand": stroke_expand,
        "enable_outline": enable_outline,
        "outline_color": outline_color,
        "outline_width": outline_width,
        "enable_shadow": enable_shadow,
        "shadow_blur": shadow_blur,
        "bg_color": bg_color
    }

    # --- STEP 4: COMPOSITE & EXPORT ---
    st.markdown("---")
    
    if preset_use_case == "Etsy Commercial Alphabet Pack (A-Z Exporter)":
        st.header("Step 4: Generate & Export Complete A–Z Digital Asset Pack")
        if st.button("📦 Export Full 26-Letter Commercial PNG Pack (ZIP)", type="primary", use_container_width=True):
            with st.spinner("Rendering full 300 DPI A-Z alphabet asset pack..."):
                zip_buffer = io.BytesIO()
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for char in alphabet:
                        char_canvas = render_typography_artwork(
                            phrase=char,
                            font=active_font,
                            drawings=st.session_state["drawings"],
                            mapping_mode="Sentence-wide",
                            letter_configs={},
                            global_cfg=global_cfg,
                            styling={**styling_opts, "bg_color": "TRANSPARENT"}
                        )
                        img_buf = io.BytesIO()
                        char_canvas.save(img_buf, format="PNG", dpi=(300, 300))
                        zip_file.writestr(f"Letters/{char}_Commercial_Artwork.png", img_buf.getvalue())
                        
                st.success("A–Z Commercial Alphabet Pack successfully generated!")
                st.download_button(
                    label="📥 Download Full A–Z Digital Asset Pack (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="Etsy_Commercial_Alphabet_Pack.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    else:
        st.header("Step 4: Render Artwork & Preview Product Mockups")
        if st.button("🚀 Render High-Res Typography Artwork & Mockups", type="primary", use_container_width=True):
            if not phrase.strip():
                st.warning("Please enter a text phrase.")
            else:
                with st.spinner("Rendering commercial artwork and product mockup..."):
                    final_canvas = render_typography_artwork(
                        phrase=phrase,
                        font=active_font,
                        drawings=st.session_state["drawings"],
                        mapping_mode=mapping_mode,
                        letter_configs=letter_configs,
                        global_cfg=global_cfg,
                        styling=styling_opts
                    )

                    mockup_img = generate_product_mockup(final_canvas, mockup_choice)

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.subheader("🖼️ High-Res Typography Output")
                        st.image(final_canvas, use_container_width=True)
                    with res_col2:
                        st.subheader(f"👕 Live {mockup_choice} Merchandise Preview")
                        st.image(mockup_img, use_container_width=True)

                    # Downloads
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        buf1 = io.BytesIO()
                        final_canvas.save(buf1, format="PNG", dpi=(300, 300))
                        st.download_button(
                            label="📥 Download Printable Artwork (PNG)",
                            data=buf1.getvalue(),
                            file_name=f"{phrase}_Typography_Artwork.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    with dl_col2:
                        buf2 = io.BytesIO()
                        mockup_img.save(buf2, format="PNG")
                        st.download_button(
                            label=f"📥 Download {mockup_choice} Product Mockup",
                            data=buf2.getvalue(),
                            file_name=f"{phrase}_{mockup_choice}_Mockup.png",
                            mime="image/png",
                            use_container_width=True
                        )
