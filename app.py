import io
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Page configuration
st.set_page_config(
    page_title="Solid Typed Artwork Studio",
    page_icon="🎨",
    layout="wide"
)

MAX_SIZE = 1800

# ============================================================
# RELIABLE BOLD FONT ENGINE
# ============================================================

FONT_OPTIONS = {
    "Heavy Impact Block": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf"
    ],
    "Classic Sans Bold": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf"
    ],
    "Serif Bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "C:/Windows/Fonts/georgiab.ttf"
    ]
}

def load_bold_font(font_choice, size):
    paths = FONT_OPTIONS.get(font_choice, [])
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

# ============================================================
# COMPOSITING ENGINE
# ============================================================

def resize_image(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img.copy()
    scale = max_size / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def render_giant_solid_typed_mask(w, h, lines, font_choice, base_font_size, text_angle):
    """Renders giant, thick, solid typed letters on a blank mask."""
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)

    active_lines = [l.strip() for l in lines if l.strip()]
    if not active_lines:
        return np.array(mask_img)

    font_obj = load_bold_font(font_choice, base_font_size)

    # Calculate positioning for stacked lines
    line_bboxes = [draw.textbbox((0, 0), line, font=font_obj) for line in active_lines]
    line_heights = [max(30, b[3] - b[1]) for b in line_bboxes]
    line_spacing = int(base_font_size * 0.25)
    
    total_height = sum(line_heights) + (len(active_lines) - 1) * line_spacing
    start_y = (h - total_height) // 2

    for idx, line in enumerate(active_lines):
        bbox = line_bboxes[idx]
        tw = bbox[2] - bbox[0]
        tx = (w - tw) // 2 - bbox[0]
        ty = start_y - bbox[1]
        
        # Draw solid thick typed text
        draw.text((tx, ty), line, font=font_obj, fill=255)
        start_y += line_heights[idx] + line_spacing

    # Apply angle rotation if needed
    if text_angle != 0:
        mask_img = mask_img.rotate(text_angle, resample=Image.Resampling.BICUBIC, expand=False)

    return np.array(mask_img)

def extract_outer_frame_mask(style_rgb_img):
    """Extracts long outer lines using morphological open/close operations."""
    arr = np.array(style_rgb_img).astype(np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 12
    )
    
    # Large structural kernel to filter out small handwritten strokes and keep outer frame lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def composite_artwork_into_typed_design(
    artwork_img, text_style_img, lines, font_choice, font_size, text_angle, include_frame
):
    art_rgb = artwork_img.convert("RGB")
    style_rgb = text_style_img.convert("RGB")
    
    style_rgb = resize_image(style_rgb, MAX_SIZE)
    w, h = style_rgb.size
    art_rgb = art_rgb.resize((w, h), Image.Resampling.LANCZOS)
    
    # 1. Create SOLID TYPED MASK (This guarantees full letter fills)
    typed_mask = render_giant_solid_typed_mask(
        w=w,
        h=h,
        lines=lines,
        font_choice=font_choice,
        base_font_size=font_size,
        text_angle=text_angle
    )

    # 2. Add outer frame if selected
    if include_frame:
        frame_mask = extract_outer_frame_mask(style_rgb)
        final_mask = cv2.bitwise_or(typed_mask, frame_mask)
    else:
        final_mask = typed_mask

    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

    # 3. Clip Slot 1 Artwork Pattern into the solid mask
    art_arr = np.array(art_rgb)
    rgba_arr = np.dstack((art_arr, final_mask))
    composited = Image.fromarray(rgba_arr, "RGBA")

    bbox = composited.getbbox()
    if bbox:
        composited = composited.crop(bbox)

    padding = 40
    padded = Image.new("RGBA", (composited.width + padding * 2, composited.height + padding * 2), (0, 0, 0, 0))
    padded.alpha_composite(composited, (padding, padding))

    return padded

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
# STREAMLIT INTERFACE
# ============================================================

st.title("🎨 Solid Typed Artwork Studio")
st.write("Convert hand-drawn layout sketches into **bold, fully filled typed letters** using your artwork pattern!")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.header("Slot 1: Artwork / Pattern Fill")
    art_file = st.file_uploader(
        "Upload Artwork / Painting photo (JPG, PNG):",
        type=["jpg", "jpeg", "png", "webp"],
        key="slot1_art"
    )
    if art_file:
        img1 = Image.open(art_file)
        img1 = ImageOps.exif_transpose(img1)
        st.image(img1, caption="Uploaded Artwork Pattern", use_container_width=True)

with col_upload2:
    st.header("Slot 2: Text Layout Sketch")
    style_file = st.file_uploader(
        "Upload Hand-Drawn Layout Sketch photo (JPG, PNG):",
        type=["jpg", "jpeg", "png", "webp"],
        key="slot2_style"
    )
    if style_file:
        img2 = Image.open(style_file)
        img2 = ImageOps.exif_transpose(img2)
        st.image(img2, caption="Uploaded Sketch Layout", use_container_width=True)

st.markdown("---")
st.header("Step 3: Enter Typed Sentence & Layout Options")

col_line1, col_line2, col_line3 = st.columns(3)
with col_line1:
    line1 = st.text_input("Line 1 Text:", "I AM").upper()
with col_line2:
    line2 = st.text_input("Line 2 Text:", "NOT").upper()
with col_line3:
    line3 = st.text_input("Line 3 Text:", "TRASH!").upper()

col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
with col_opt1:
    font_choice = st.selectbox("Typed Font Style:", list(FONT_OPTIONS.keys()))
with col_opt2:
    font_size = st.slider("Typed Font Size (px):", 100, 450, 240)
with col_opt3:
    text_angle = st.slider("Text Tilt Angle (°):", -45, 45, -10)
with col_opt4:
    include_frame = st.checkbox("Include Outer Frame", value=False)

mockup_choice = st.selectbox("Select 3D Merchandise Mockup:", ["Men's Classic Crew Neck T-Shirt", "Boutique Tote Bag"])

if st.button("🚀 Render Solid Typed Artwork Design", type="primary", use_container_width=True):
    if not art_file or not style_file:
        st.warning("Please upload BOTH Slot 1 (Artwork) and Slot 2 (Sketch Layout) images.")
    else:
        with st.spinner("Generating solid typed letters and filling with artwork..."):
            art_img = Image.open(art_file)
            art_img = ImageOps.exif_transpose(art_img)
            
            style_img = Image.open(style_file)
            style_img = ImageOps.exif_transpose(style_img)

            composited_result = composite_artwork_into_typed_design(
                artwork_img=art_img,
                text_style_img=style_img,
                lines=[line1, line2, line3],
                font_choice=font_choice,
                font_size=font_size,
                text_angle=text_angle,
                include_frame=include_frame
            )

            mockup_img = generate_3d_product_mockup(composited_result, mockup_choice)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("🖼️ Solid Typed Artwork Design")
                st.image(composited_result, use_container_width=True)
            with res_col2:
                st.subheader(f"👕 Live 3D {mockup_choice} Mockup")
                st.image(mockup_img, use_container_width=True)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                buf1 = io.BytesIO()
                composited_result.save(buf1, format="PNG", dpi=(300, 300))
                st.download_button(
                    label="📥 Download Transparent Composited PNG",
                    data=buf1.getvalue(),
                    file_name="Solid_Typed_Artwork_Design.png",
                    mime="image/png",
                    use_container_width=True
                )
            with dl_col2:
                buf2 = io.BytesIO()
                mockup_img.save(buf2, format="PNG")
                st.download_button(
                    label="📥 Download 3D Mockup PNG",
                    data=buf2.getvalue(),
                    file_name="3D_Merch_Mockup.png",
                    mime="image/png",
                    use_container_width=True
                )
