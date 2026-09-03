import io
import os
import random
import math
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

# Page Configuration
st.set_page_config(
    page_title="Hand-Drawn Artwork FX Studio",
    page_icon="🎨",
    layout="wide"
)

MAX_SIZE = 1800

# ============================================================
# FONT HELPER
# ============================================================

def get_font(size, bold=True):
    font_names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    ]
    for font_path in font_names:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

# ============================================================
# BACKGROUND REMOVAL & EXTRACTION ALGORITHM
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
# CUTE FX ENGINES
# ============================================================

def apply_sticker_outline(img: Image.Image, border_color=(255, 255, 255), border_size=12) -> Image.Image:
    """Creates a cute die-cut vinyl sticker effect with a thick white outline."""
    alpha = img.getchannel("A")
    dilated_alpha = alpha.filter(ImageFilter.MaxFilter(border_size * 2 + 1))
    
    # Create sticker background layer
    sticker_bg = Image.new("RGBA", img.size, border_color + (0,))
    sticker_bg.putalpha(dilated_alpha)

    # Shadow for sticker
    shadow_alpha = dilated_alpha.filter(ImageFilter.GaussianBlur(8))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha.point(lambda p: int(p * 0.3)))
    
    output = Image.new("RGBA", (img.width + 20, img.height + 20), (0, 0, 0, 0))
    output.alpha_composite(shadow, (10, 14))
    output.alpha_composite(sticker_bg, (10, 10))
    output.alpha_composite(img, (10, 10))
    return output

def apply_pastel_watercolor_glow(img: Image.Image, glow_color=(255, 182, 193)) -> Image.Image:
    """Adds a dreamy pastel watercolor aura around the object."""
    alpha = img.getchannel("A")
    glow_alpha = alpha.filter(ImageFilter.GaussianBlur(25))
    
    glow_layer = Image.new("RGBA", img.size, glow_color + (0,))
    glow_layer.putalpha(glow_alpha.point(lambda p: int(p * 0.7)))
    
    output = Image.new("RGBA", img.size, (0, 0, 0, 0))
    output.alpha_composite(glow_layer, (0, 0))
    output.alpha_composite(img, (0, 0))
    return output

def apply_pencil_sketch(img: Image.Image) -> Image.Image:
    """Converts image or object to a soft pencil sketch."""
    rgb = img.convert("RGB")
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    
    sketch_rgb = Image.fromarray(sketch).convert("RGBA")
    sketch_rgb.putalpha(img.getchannel("A"))
    return sketch_rgb

def create_text_image_mask(text: str, fill_image: Image.Image, font_size: int = 150) -> Image.Image:
    """Clips the hand-drawn object/texture into custom rendered text."""
    font = get_font(font_size, bold=True)
    
    # Measure text box size
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    tw = max(10, bbox[2] - bbox[0] + 40)
    th = max(10, bbox[3] - bbox[1] + 40)
    
    # Render text mask
    mask_img = Image.new("L", (tw, th), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.text((20, 20), text, font=font, fill=255)
    
    # Tile/Resize hand-drawn image inside text
    tiled_fill = fill_image.resize((tw, th), Image.Resampling.LANCZOS).convert("RGBA")
    tiled_fill.putalpha(mask_img)
    return tiled_fill

# ============================================================
# STREAMLIT UI & WORKFLOW
# ============================================================

st.title("🎨 Hand-Drawn Object Mixer & FX Studio")
st.write("Upload photos or hand-drawn sketches, extract distinct objects, mix them onto a dynamic canvas, apply cute FX, or mask your drawings directly inside text!")

if "extracted_objects" not in st.session_state:
    st.session_state["extracted_objects"] = []

# --- STEP 1: UPLOAD & EXTRACT COMPONENTS ---
st.header("Step 1: Upload Photos & Extract Components")

uploaded_files = st.file_uploader(
    "Upload Hand-Drawn Sketches or Photos (JPG, PNG):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if st.button("✂️ Extract All Objects & Isolate Backgrounds", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        extracted = []
        with st.spinner("Extracting objects from backgrounds..."):
            for f in uploaded_files:
                raw = Image.open(f)
                raw = ImageOps.exif_transpose(raw).convert("RGB")
                resized = resize_image(raw, MAX_SIZE)
                isolated = create_transparent_drawing(resized)
                extracted.append({"name": f.name, "image": isolated})
        st.session_state["extracted_objects"] = extracted
        st.success(f"Isolated {len(extracted)} component(s) successfully!")

# --- STEP 2: DISPLAY EXTRACTED OBJECTS & FX PRESETS ---
if st.session_state["extracted_objects"]:
    st.markdown("---")
    st.header("Step 2: Component Library & Cute FX Styles")
    
    cols = st.columns(min(4, len(st.session_state["extracted_objects"])))
    processed_objects = []

    for idx, item in enumerate(st.session_state["extracted_objects"]):
        with cols[idx % len(cols)]:
            st.markdown(f"**Component #{idx + 1}** ({item['name']})")
            st.image(item["image"], use_container_width=True)
            
            fx_choice = st.selectbox(
                f"Apply Effect to #{idx + 1}:",
                ["Original Isolated", "Cute Sticker Outline", "Pastel Glow", "Pencil Sketch"],
                key=f"fx_{idx}"
            )
            
            obj_img = item["image"].copy()
            if fx_choice == "Cute Sticker Outline":
                obj_img = apply_sticker_outline(obj_img)
            elif fx_choice == "Pastel Glow":
                obj_img = apply_pastel_watercolor_glow(obj_img)
            elif fx_choice == "Pencil Sketch":
                obj_img = apply_pencil_sketch(obj_img)
                
            processed_objects.append({"idx": idx, "name": f"Object #{idx + 1}", "image": obj_img})

    # --- STEP 3: TEXT INSPIRATION & CLIPPING MASK ---
    st.markdown("---")
    st.header("Step 3: Text Masking (Incorporate Drawing Inside Words)")
    
    enable_text_mask = st.checkbox("Incorporate hand-drawn drawing inside custom Text/Word?")
    text_masked_img = None
    
    if enable_text_mask:
        tm_col1, tm_col2, tm_col3 = st.columns([2, 2, 1])
        with tm_col1:
            custom_text = st.text_input("Text/Word to Display:", "CREATE")
        with tm_col2:
            fill_obj_idx = st.selectbox(
                "Select Drawing/Texture to fill inside text:",
                options=range(len(processed_objects)),
                format_func=lambda x: f"Object #{x + 1}"
            )
        with tm_col3:
            font_size = st.slider("Font Size:", 80, 300, 160)
            
        if custom_text.strip():
            text_masked_img = create_text_image_mask(
                custom_text, 
                processed_objects[fill_obj_idx]["image"], 
                font_size=font_size
            )
            st.image(text_masked_img, caption="Text Masking Preview", width=400)

    # --- STEP 4: CANVAS COMPOSITOR ---
    st.markdown("---")
    st.header("Step 4: Composite & Mix Selected Components")

    canvas_bg_color = st.color_picker("Canvas Background Color:", "#F0F4F8")
    canvas_w = st.number_input("Canvas Width (px):", 800, 3000, 1200)
    canvas_h = st.number_input("Canvas Height (px):", 600, 3000, 900)

    selected_indices = st.multiselect(
        "Select Components to mix onto Canvas:",
        options=list(range(len(processed_objects))),
        default=list(range(len(processed_objects))),
        format_func=lambda x: f"Object #{x + 1}"
    )

    # Individual Layout Controls for chosen components
    placements = []
    if selected_indices:
        st.subheader("Position & Transform Selected Components")
        for sel_idx in selected_indices:
            with st.expander(f"Transform Controls for Object #{sel_idx + 1}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    pos_x = st.slider(f"X Position #{sel_idx + 1}:", 0, canvas_w, 100 + (sel_idx * 120), key=f"x_{sel_idx}")
                with c2:
                    pos_y = st.slider(f"Y Position #{sel_idx + 1}:", 0, canvas_h, 100 + (sel_idx * 80), key=f"y_{sel_idx}")
                with c3:
                    scale = st.slider(f"Scale #{sel_idx + 1}:", 0.1, 2.0, 0.6, key=f"scale_{sel_idx}")
                with c4:
                    rotation = st.slider(f"Rotation #{sel_idx + 1}:", -180, 180, 0, key=f"rot_{sel_idx}")
                
                placements.append({
                    "image": processed_objects[sel_idx]["image"],
                    "x": pos_x, "y": pos_y, "scale": scale, "rotation": rotation
                })

    # Render Final Canvas
    if st.button("🚀 Render Mixed Composition Canvas", type="primary", use_container_width=True):
        # Base canvas
        bg_rgb = tuple(int(canvas_bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_rgb + (255,))
        
        # Place Text Mask if active
        if enable_text_mask and text_masked_img:
            tx = (canvas_w - text_masked_img.width) // 2
            ty = int(canvas_h * 0.1)
            canvas.alpha_composite(text_masked_img, (tx, ty))

        # Composite selected objects
        for p in placements:
            img = p["image"].copy()
            
            # Apply Rotation
            if p["rotation"] != 0:
                img = img.rotate(-p["rotation"], expand=True, resample=Image.Resampling.BICUBIC)
                
            # Apply Scale
            nw, nh = max(10, int(img.width * p["scale"])), max(10, int(img.height * p["scale"]))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
            
            # Composite onto canvas
            canvas.alpha_composite(img, (p["x"], p["y"]))

        st.subheader("🖼️ Final Mixed Artwork Composition")
        st.image(canvas.convert("RGB"), use_container_width=True)

        # Download Result
        buf = io.BytesIO()
        canvas.convert("RGB").save(buf, format="PNG")
        st.download_button(
            label="📥 Download Mixed High-Res Image (PNG)",
            data=buf.getvalue(),
            file_name="mixed_handdrawn_composition.png",
            mime="image/png",
            use_container_width=True
        )
