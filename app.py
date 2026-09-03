import io
import os
import math
import random
import zipfile
import urllib.request
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Page Configuration
st.set_page_config(
    page_title="T-Shirt Art Typography Generator",
    page_icon="👕",
    layout="wide"
)

FONT_DIR = "./fonts"
os.makedirs(FONT_DIR, exist_ok=True)

# ============================================================
# BUILT-IN GOOGLE FONTS & LOADER
# ============================================================

DEFAULT_FONTS = {
    "Bebas Neue": "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Pacifico": "https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf",
    "Lobster": "https://github.com/google/fonts/raw/main/ofl/lobster/Lobster-Regular.ttf",
    "Montserrat Bold": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
    "Anton": "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
}

def ensure_default_fonts():
    for name, url in DEFAULT_FONTS.items():
        path = os.path.join(FONT_DIR, f"{name.replace(' ', '_')}.ttf")
        if not os.path.exists(path):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as resp, open(path, 'wb') as f:
                    f.write(resp.read())
            except Exception:
                pass

def get_font_list(custom_fonts=None):
    ensure_default_fonts()
    font_files = {}
    if os.path.exists(FONT_DIR):
        for f in os.listdir(FONT_DIR):
            if f.lower().endswith(('.ttf', '.otf')):
                font_files[os.path.splitext(f)[0].replace('_', ' ')] = os.path.join(FONT_DIR, f)
    if custom_fonts:
        font_files.update(custom_fonts)
    return font_files

def load_font(font_path, size):
    try:
        return ImageFont.truetype(font_path, size=int(size))
    except Exception:
        return ImageFont.load_default()

def measure_text(text, font):
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h, bbox

# ============================================================
# ARTWORK PREPARATION ENGINE
# ============================================================

def prepare_artwork_layer(artworks, canvas_size, seed=42):
    random.seed(seed)
    w_canvas, h_canvas = canvas_size
    
    if not artworks:
        arr = np.zeros((h_canvas, w_canvas, 4), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(255, 0, h_canvas)[:, None]
        arr[:, :, 1] = np.linspace(0, 255, w_canvas)[None, :]
        arr[:, :, 2] = 200
        arr[:, :, 3] = 255
        return Image.fromarray(arr, "RGBA")

    chosen_art = random.choice(artworks).copy().convert("RGBA")
    transform_type = random.choice(["stretch_fit", "tile", "center_zoom", "mirror_tile", "random_crop"])
    
    if transform_type == "stretch_fit":
        return chosen_art.resize((w_canvas, h_canvas), Image.Resampling.LANCZOS)
    elif transform_type == "center_zoom":
        scale = random.uniform(1.2, 2.5)
        nw, nh = int(w_canvas * scale), int(h_canvas * scale)
        resized = chosen_art.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - w_canvas) // 2
        top = (nh - h_canvas) // 2
        return resized.crop((left, top, left + w_canvas, top + h_canvas))
    elif transform_type in ["tile", "mirror_tile"]:
        tile_w = random.randint(w_canvas // 5, w_canvas // 2)
        aspect = chosen_art.height / max(1, chosen_art.width)
        tile_h = max(10, int(tile_w * aspect))
        tile_img = chosen_art.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        if transform_type == "mirror_tile":
            tile_img = ImageOps.mirror(tile_img)
        canvas_art = Image.new("RGBA", (w_canvas, h_canvas), (0, 0, 0, 0))
        for x in range(0, w_canvas, tile_w):
            for y in range(0, h_canvas, tile_h):
                canvas_art.alpha_composite(tile_img, (x, y))
        return canvas_art
    else:
        w_art, h_art = chosen_art.size
        crop_w = int(w_art * random.uniform(0.5, 0.9))
        crop_h = int(h_art * random.uniform(0.5, 0.9))
        left = random.randint(0, max(0, w_art - crop_w))
        top = random.randint(0, max(0, h_art - crop_h))
        cropped = chosen_art.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w_canvas, h_canvas), Image.Resampling.LANCZOS)

# ============================================================
# PROCEDURAL LAYOUT GENERATOR
# ============================================================

def apply_layout_style(style_id, phrase, font_path, canvas_size, seed=42):
    random.seed(seed)
    w_canvas, h_canvas = canvas_size
    mask = Image.new("L", (w_canvas, h_canvas), 0)
    draw = ImageDraw.Draw(mask)
    
    words = phrase.strip().split()
    if not words:
        words = ["TEXT"]

    base_size = int(w_canvas / max(4, len(phrase) * 0.55))
    font = load_font(font_path, base_size)

    # Style 1: Classic Centered
    if style_id == 1:
        w, h, bbox = measure_text(phrase, font)
        draw.text(((w_canvas - w) // 2 - bbox[0], (h_canvas - h) // 2 - bbox[1]), phrase, font=font, fill=255)

    # Style 2: Stacked Multi-Line
    elif style_id == 2:
        line_heights, fonts, bboxes = [], [], []
        for word in words:
            f = load_font(font_path, int(w_canvas / max(3, len(word) * 0.7)))
            w, h, b = measure_text(word, f)
            fonts.append(f)
            line_heights.append(h)
            bboxes.append(b)

        total_h = sum(line_heights) + (len(words) - 1) * 20
        curr_y = (h_canvas - total_h) // 2

        for i, word in enumerate(words):
            w, h, b = measure_text(word, fonts[i])
            draw.text(((w_canvas - w) // 2 - b[0], curr_y - b[1]), word, font=fonts[i], fill=255)
            curr_y += h + 20

    # Style 3: Word Emphasis
    elif style_id == 3:
        main_idx = random.randint(0, len(words) - 1)
        curr_y = h_canvas // 4
        for i, word in enumerate(words):
            f_size = base_size * 2.2 if i == main_idx else base_size * 0.9
            f = load_font(font_path, f_size)
            w, h, b = measure_text(word, f)
            draw.text(((w_canvas - w) // 2 - b[0], curr_y - b[1]), word, font=f, fill=255)
            curr_y += h + 25

    # Style 4: Mixed Letter Sizes
    elif style_id == 4:
        total_w = 0
        char_imgs = []
        for char in phrase:
            f_size = base_size * random.uniform(0.7, 1.4)
            f = load_font(font_path, f_size)
            w, h, b = measure_text(char, f)
            c_img = Image.new("L", (max(10, w + 20), max(10, h + 20)), 0)
            cdraw = ImageDraw.Draw(c_img)
            cdraw.text((10 - b[0], 10 - b[1]), char, font=f, fill=255)
            char_imgs.append((c_img, w))
            total_w += w + 5

        curr_x = (w_canvas - total_w) // 2
        for c_img, w in char_imgs:
            mask.paste(c_img, (curr_x, (h_canvas - c_img.height) // 2), c_img)
            curr_x += w + 5

    # Style 5: Alternating Baseline
    elif style_id == 5:
        curr_x = int(w_canvas * 0.1)
        for i, char in enumerate(phrase):
            w, h, b = measure_text(char, font)
            y_offset = -40 if i % 2 == 0 else 40
            draw.text((curr_x - b[0], (h_canvas - h) // 2 + y_offset - b[1]), char, font=font, fill=255)
            curr_x += w + 8

    # Style 6: Rotated Letters
    elif style_id == 6:
        curr_x = int(w_canvas * 0.12)
        for char in phrase:
            w, h, b = measure_text(char, font)
            c_img = Image.new("L", (w + 40, h + 40), 0)
            cdraw = ImageDraw.Draw(c_img)
            cdraw.text((20 - b[0], 20 - b[1]), char, font=font, fill=255)
            rot = c_img.rotate(random.uniform(-18, 18), resample=Image.Resampling.BICUBIC, expand=True)
            mask.paste(rot, (curr_x, (h_canvas - rot.height) // 2), rot)
            curr_x += w + 10

    # Style 7: Wave Typography
    elif style_id == 7:
        curr_x = int(w_canvas * 0.08)
        for i, char in enumerate(phrase):
            w, h, b = measure_text(char, font)
            y_wave = int(math.sin(i * 0.5) * 60)
            draw.text((curr_x - b[0], (h_canvas - h) // 2 + y_wave - b[1]), char, font=font, fill=255)
            curr_x += w + 6

    # Style 8: Arc Typography
    elif style_id == 8:
        radius = w_canvas * 0.38
        center_x, center_y = w_canvas // 2, h_canvas // 2 + 200
        angle_spread = 110
        num_chars = len(phrase)

        for i, char in enumerate(phrase):
            if char == " ":
                continue
            angle_deg = -angle_spread / 2 + (i / max(1, num_chars - 1)) * angle_spread
            angle_rad = math.radians(angle_deg - 90)

            x = center_x + radius * math.cos(angle_rad)
            y = center_y + radius * math.sin(angle_rad)

            w, h, b = measure_text(char, font)
            c_img = Image.new("L", (w + 30, h + 30), 0)
            cdraw = ImageDraw.Draw(c_img)
            cdraw.text((15 - b[0], 15 - b[1]), char, font=font, fill=255)
            rot = c_img.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
            mask.paste(rot, (int(x - rot.width // 2), int(y - rot.height // 2)), rot)

    # Style 9: Circular Composition
    elif style_id == 9:
        radius = w_canvas * 0.32
        center_x, center_y = w_canvas // 2, h_canvas // 2
        num_chars = len(phrase)

        for i, char in enumerate(phrase):
            angle_deg = (i / max(1, num_chars)) * 360
            angle_rad = math.radians(angle_deg - 90)

            x = center_x + radius * math.cos(angle_rad)
            y = center_y + radius * math.sin(angle_rad)

            w, h, b = measure_text(char, font)
            c_img = Image.new("L", (w + 20, h + 20), 0)
            cdraw = ImageDraw.Draw(c_img)
            cdraw.text((10 - b[0], 10 - b[1]), char, font=font, fill=255)
            rot = c_img.rotate(-angle_deg + 90, resample=Image.Resampling.BICUBIC, expand=True)
            mask.paste(rot, (int(x - rot.width // 2), int(y - rot.height // 2)), rot)

    # Style 10: Vertical Composition
    elif style_id == 10:
        curr_y = int(h_canvas * 0.1)
        for char in phrase:
            w, h, b = measure_text(char, font)
            draw.text(((w_canvas - w) // 2 - b[0], curr_y - b[1]), char, font=font, fill=255)
            curr_y += h + 12

    # Style 11+: Compound Layout Fallbacks
    else:
        curr_y = int(h_canvas * 0.2)
        for word in words:
            w, h, b = measure_text(word, font)
            draw.text(((w_canvas - w) // 2 - b[0], curr_y - b[1]), word, font=font, fill=255)
            curr_y += h + random.randint(15, 60)

    # Fit Boundary Safeguard
    bbox = mask.getbbox()
    if bbox:
        mw = bbox[2] - bbox[0]
        mh = bbox[3] - bbox[1]
        if mw > w_canvas * 0.92 or mh > h_canvas * 0.92:
            scale = min((w_canvas * 0.88) / mw, (h_canvas * 0.88) / mh)
            cropped = mask.crop(bbox)
            new_size = (max(1, int(mw * scale)), max(1, int(mh * scale)))
            resized_mask = cropped.resize(new_size, Image.Resampling.LANCZOS)
            
            mask = Image.new("L", (w_canvas, h_canvas), 0)
            mask.paste(resized_mask, ((w_canvas - new_size[0]) // 2, (h_canvas - new_size[1]) // 2))

    return mask

# ============================================================
# CLIPPING MASK & EFFECTS
# ============================================================

def clip_artwork_to_mask(artwork_layer, text_mask):
    art_rgba = artwork_layer.convert("RGBA")
    mask_l = text_mask.convert("L")

    art_arr = np.array(art_rgba, dtype=np.uint8)
    mask_arr = np.array(mask_l, dtype=np.uint8)

    art_alpha = art_arr[:, :, 3].astype(np.float32) / 255.0
    text_alpha = mask_arr.astype(np.float32) / 255.0
    
    final_alpha = (art_alpha * text_alpha * 255.0).astype(np.uint8)
    art_arr[:, :, 3] = final_alpha
    
    return Image.fromarray(art_arr, "RGBA")

def apply_text_effects(clipped_image, text_mask, enable_stroke=True, stroke_width=12, enable_shadow=False, shadow_offset=(18, 18)):
    w_canvas, h_canvas = text_mask.size
    mask_arr = np.array(text_mask.convert("L"), dtype=np.uint8)
    
    final_canvas = Image.new("RGBA", (w_canvas, h_canvas), (0, 0, 0, 0))

    if enable_shadow:
        shadow_arr = cv2.dilate(mask_arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
        shadow_rgba = np.zeros((h_canvas, w_canvas, 4), dtype=np.uint8)
        shadow_rgba[:, :, 3] = (shadow_arr * 0.45).astype(np.uint8)
        final_canvas.alpha_composite(Image.fromarray(shadow_rgba, "RGBA"), shadow_offset)

    if enable_stroke and stroke_width > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width * 2 + 1, stroke_width * 2 + 1))
        dilated_mask = cv2.dilate(mask_arr, kernel, iterations=1)
        border_mask = cv2.subtract(dilated_mask, mask_arr)

        border_rgba = np.zeros((h_canvas, w_canvas, 4), dtype=np.uint8)
        border_rgba[:, :, 3] = border_mask
        final_canvas.alpha_composite(Image.fromarray(border_rgba, "RGBA"))

    final_canvas.alpha_composite(clipped_image)
    return final_canvas

def generate_single_design(design_id, phrase, artworks, fonts_map, canvas_size=(3000, 3000), master_seed=12345, enable_stroke=True, enable_shadow=False):
    seed = master_seed + (design_id * 777)
    random.seed(seed)

    layout_style = (design_id % 15) + 1
    font_name = random.choice(list(fonts_map.keys()))
    font_path = fonts_map[font_name]

    text_mask = apply_layout_style(layout_style, phrase, font_path, canvas_size, seed)
    artwork_layer = prepare_artwork_layer(artworks, canvas_size, seed)
    clipped = clip_artwork_to_mask(artwork_layer, text_mask)

    return apply_text_effects(clipped, text_mask, enable_stroke, random.randint(8, 20), enable_shadow)

# ============================================================
# EXPORT & CONTACT SHEET UTILITIES
# ============================================================

def create_contact_sheet(designs, cols=5, thumb_size=(300, 300)):
    rows = (len(designs) + cols - 1) // cols
    sheet_w = cols * thumb_size[0] + (cols + 1) * 20
    sheet_h = rows * thumb_size[1] + (rows + 1) * 40

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (240, 242, 245, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, img in enumerate(designs):
        r, c = idx // cols, idx % cols
        x = c * thumb_size[0] + (c + 1) * 20
        y = r * thumb_size[1] + (r + 1) * 40

        draw.rectangle([x - 5, y - 5, x + thumb_size[0] + 5, y + thumb_size[1] + 25], fill=(255, 255, 255, 255), outline=(210, 215, 220), width=2)
        thumb = img.copy()
        thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        
        tx = x + (thumb_size[0] - thumb.width) // 2
        ty = y + (thumb_size[1] - thumb.height) // 2
        sheet.alpha_composite(thumb, (tx, ty))
        draw.text((x + 10, y + thumb_size[1] + 5), f"Design #{idx+1:02d}", fill=(40, 40, 40))

    return sheet

def export_zip_archive(designs):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img in enumerate(designs):
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG", dpi=(300, 300))
            zip_file.writestr(f"design_{idx+1:03d}.png", img_buf.getvalue())
    return zip_buf.getvalue()

def render_tshirt_mockup(design_img, shirt_color=(245, 245, 245)):
    canvas_w, canvas_h = 1000, 1000
    mockup = Image.new("RGBA", (canvas_w, canvas_h), (235, 238, 242, 255))
    draw = ImageDraw.Draw(mockup)

    shirt_pts = [
        (300, 120), (400, 80), (600, 80), (700, 120),
        (880, 280), (780, 400), (720, 350), (720, 920),
        (280, 920), (280, 350), (220, 400), (120, 280)
    ]
    draw.polygon(shirt_pts, fill=shirt_color, outline=(180, 185, 190), width=4)
    draw.arc((400, 60, 600, 160), start=0, end=180, fill=(160, 165, 170), width=5)

    target_w = 380
    scale = target_w / float(design_img.width)
    target_h = int(design_img.height * scale)
    
    resized_art = design_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mockup.alpha_composite(resized_art, ((canvas_w - target_w) // 2, 280))

    return mockup

# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

if "generated_designs" not in st.session_state:
    st.session_state.generated_designs = []
if "custom_fonts" not in st.session_state:
    st.session_state.custom_fonts = {}

st.sidebar.title("🎨 Generator Settings")
master_seed = st.sidebar.number_input("Random Seed:", value=12345, step=1)
canvas_choice = st.sidebar.selectbox("Canvas Resolution:", ["3000 x 3000 px (Standard)", "4000 x 4000 px (Ultra HD)"])
canvas_dim = 3000 if "3000" in canvas_choice else 4000
num_designs = st.sidebar.slider("Number of Designs:", min_value=1, max_value=100, value=50)

st.sidebar.subheader("✨ Text Effects")
enable_stroke = st.sidebar.checkbox("Add Outer Black Border", value=True)
enable_shadow = st.sidebar.checkbox("Add Drop Shadow", value=False)

uploaded_fonts = st.sidebar.file_uploader("Upload Custom .TTF / .OTF Fonts:", type=["ttf", "otf"], accept_multiple_files=True)
if uploaded_fonts:
    for font_file in uploaded_fonts:
        fpath = os.path.join(FONT_DIR, font_file.name)
        with open(fpath, "wb") as f:
            f.write(font_file.read())
        st.session_state.custom_fonts[os.path.splitext(font_file.name)[0]] = fpath

fonts_map = get_font_list(custom_fonts=st.session_state.custom_fonts)

st.title("👕 T-Shirt Art Typography Generator")
st.write("Generate 50 high-resolution, print-ready T-shirt designs with uploaded artwork clipped inside custom procedural typography.")

tab1, tab2, tab3 = st.tabs(["🚀 Generator", "🖼️ Gallery & Contact Sheet", "👕 3D Mockup Preview"])

with tab1:
    col_art, col_text = st.columns([1, 1])

    with col_art:
        st.subheader("1. Upload Artwork")
        art_files = st.file_uploader("Upload Artwork Images (PNG, JPG, WEBP):", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        artworks = []
        if art_files:
            cols = st.columns(min(4, len(art_files)))
            for idx, file in enumerate(art_files):
                img = Image.open(file).convert("RGBA")
                artworks.append(img)
                with cols[idx % 4]:
                    st.image(img, caption=f"Art #{idx+1}", use_container_width=True)

    with col_text:
        st.subheader("2. Text Phrase")
        phrase = st.text_area("Enter Sentence or Phrase:", value="MAKE TODAY BEAUTIFUL", height=100).upper()

    st.markdown("---")

    if st.button("🚀 GENERATE 50 DESIGNS", type="primary", use_container_width=True):
        if not phrase.strip():
            st.error("Please enter a sentence or phrase.")
        else:
            st.session_state.generated_designs = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(num_designs):
                status_text.text(f"Generating Design #{i+1:02d} of {num_designs}...")
                design = generate_single_design(
                    design_id=i,
                    phrase=phrase,
                    artworks=artworks,
                    fonts_map=fonts_map,
                    canvas_size=(canvas_dim, canvas_dim),
                    master_seed=master_seed,
                    enable_stroke=enable_stroke,
                    enable_shadow=enable_shadow
                )
                st.session_state.generated_designs.append(design)
                progress_bar.progress((i + 1) / num_designs)

            status_text.success(f"Successfully generated {num_designs} designs!")

with tab2:
    if st.session_state.generated_designs:
        st.subheader("Contact Sheet Overview")
        contact_sheet = create_contact_sheet(st.session_state.generated_designs)
        st.image(contact_sheet, use_container_width=True)

        st.markdown("---")
        st.subheader("Individual Design Inspector")
        selected_idx = st.selectbox(
            "Select Design to Inspect / Download:",
            options=list(range(len(st.session_state.generated_designs))),
            format_func=lambda x: f"Design #{x+1:02d}"
        )

        sel_img = st.session_state.generated_designs[selected_idx]
        st.image(sel_img, width=500, caption=f"Design #{selected_idx+1:02d}")

        buf = io.BytesIO()
        sel_img.save(buf, format="PNG", dpi=(300, 300))
        st.download_button(
            label=f"📥 Download Design #{selected_idx+1:02d} PNG",
            data=buf.getvalue(),
            file_name=f"tshirt_design_{selected_idx+1:03d}.png",
            mime="image/png"
        )

        st.markdown("---")
        zip_bytes = export_zip_archive(st.session_state.generated_designs)
        st.download_button(
            label=f"📦 Download All {len(st.session_state.generated_designs)} Designs as ZIP Archive (300 DPI)",
            data=zip_bytes,
            file_name="tshirt_designs_pack.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("Click 'GENERATE 50 DESIGNS' in the Generator tab to preview outputs.")

with tab3:
    if st.session_state.generated_designs:
        st.subheader("T-Shirt Mockup Preview")
        shirt_color_hex = st.color_picker("T-Shirt Color:", "#F5F5F5")
        hex_val = shirt_color_hex.lstrip('#')
        rgb_color = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))

        selected_mockup_idx = st.selectbox(
            "Select Design for Mockup:",
            options=list(range(len(st.session_state.generated_designs))),
            format_func=lambda x: f"Design #{x+1:02d}",
            key="mockup_select"
        )

        mockup_img = render_tshirt_mockup(
            st.session_state.generated_designs[selected_mockup_idx],
            shirt_color=rgb_color
        )
        st.image(mockup_img, width=600)
    else:
        st.info("Generate designs first to preview mockups.")
