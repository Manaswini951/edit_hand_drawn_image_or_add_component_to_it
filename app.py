import io
import os
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Attempt importing EasyOCR; fall back to bounding box heuristic if not installed
try:
    import easyocr
    READER = easyocr.Reader(['en'], gpu=False)
    HAS_OCR = True
except Exception:
    HAS_OCR = False

# Page configuration
st.set_page_config(
    page_title="Handwriting-to-Font Artwork Studio",
    page_icon="🎨",
    layout="wide"
)

MAX_SIZE = 1800

# ============================================================
# RELIABLE FONT LOADING ENGINE
# ============================================================

def load_bold_digital_font(size):
    """Loads a thick TrueType font to ensure solid artwork fills."""
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

# ============================================================
# OCR & BOUNDING-BOX LAYOUT EXTRACTION
# ============================================================

def resize_image(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img.copy()
    scale = max_size / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def extract_framing_lines_mask(gray_img):
    """Isolates non-text surroundings like box frames or decorative ribbons."""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 12
    )
    # Filter out small dots
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    clean_mask = np.zeros_like(thresh)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= 40:
            clean_mask[labels == label] = 255
    return clean_mask

def convert_sketch_to_digital_font_mask(style_rgb_img, fallback_text="I AM NOT TRASH"):
    """
    Detects handwritten text boxes using OCR or contour segmentation, 
    then renders solid digital typography in those exact positions.
    """
    arr = np.array(style_rgb_img).astype(np.uint8)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Base canvas for digital text mask
    digital_text_mask = np.zeros((h, w), dtype=np.uint8)
    pil_mask = Image.fromarray(digital_text_mask)
    draw = ImageDraw.Draw(pil_mask)

    text_blocks = []

    # Method 1: EasyOCR Layout Detection
    if HAS_OCR:
        try:
            results = READER.readtext(arr)
            for (bbox_pts, detected_text, prob) in results:
                if prob > 0.2:
                    pts = np.array(bbox_pts, dtype=np.int32)
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    text_blocks.append({
                        "text": detected_text.upper(),
                        "bbox": (x_min, y_min, box_w, box_h)
                    })
        except Exception:
            pass

    # Method 2: Contour Bounding Boxes (Fallback if OCR yields no blocks)
    if not text_blocks:
        frame_mask = extract_framing_lines_mask(gray)
        contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        words = fallback_text.split()
        word_idx = 0
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter contours matching text proportions
            if 30 < bw < (w * 0.7) and 20 < bh < (h * 0.5) and (bw * bh) > 1000:
                text_val = words[word_idx % len(words)]
                text_blocks.append({
                    "text": text_val,
                    "bbox": (x, y, bw, bh)
                })
                word_idx += 1

    # Render Solid Digital Typography inside Detected Regions
    for block in text_blocks:
        x, y, bw, bh = block["bbox"]
        txt = block["text"]

        # Calculate optimal font size to fill the bounding box
        calc_font_size = max(24, int(bh * 0.85))
        font_obj = load_bold_digital_font(calc_font_size)
        
        # Draw solid digital text into mask
        draw.text((x + (bw // 2), y + (bh // 2)), txt, font=font_obj, fill=255, anchor="mm")

    # Merge drawn framing boxes/accents into the mask
    framing_mask = extract_framing_lines_mask(gray)
    final_combined_mask = cv2.bitwise_or(np.array(pil_mask), framing_mask)

    return final_combined_mask

# ============================================================
# COMPOSITING ENGINE: ARTWORK FILLED DIGITAL TYPOGRAPHY
# ============================================================

def composite_artwork_into_digital_typography(artwork_img, text_style_img, fallback_text):
    """Clips Upload #1 (Artwork Pattern) inside the solid digital typography mask."""
    art_rgb = artwork_img.convert("RGB")
    style_rgb = text_style_img.convert("RGB")
    
    style_rgb = resize_image(style_rgb, MAX_SIZE)
    w, h = style_rgb.size
    art_rgb = art_rgb.resize((w, h), Image.Resampling.LANCZOS)
    
    # Extract digital typography mask
    solid_mask = convert_sketch_to_digital_font_mask(style_rgb, fallback_text=fallback_text)
    solid_mask = cv2.GaussianBlur(solid_mask, (3, 3), 0)
    
    # Clip Artwork into solid mask
    art_arr = np.array(art_rgb)
    rgba_arr = np.dstack((art_arr, solid_mask))
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
# STREAMLIT TWO-SLOT INTERFACE
# ============================================================

st.title("🎨 Handwriting-to-Font Artwork Studio")
st.write("Convert handwritten text layouts into **solid digital typography** and fill them end-to-end with your artwork pattern!")

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
    st.header("Slot 2: Text Style & Layout Sketch")
    style_file = st.file_uploader(
        "Upload Hand-Drawn Writing Style / Quote Sketch photo (JPG, PNG):",
        type=["jpg", "jpeg", "png", "webp"],
        key="slot2_style"
    )
    if style_file:
        img2 = Image.open(style_file)
        img2 = ImageOps.exif_transpose(img2)
        st.image(img2, caption="Uploaded Writing Style & Layout", use_container_width=True)

st.markdown("---")
st.header("Step 3: Render Digital Typography & 3D Merch Preview")

col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    fallback_phrase = st.text_input("Verify/Enter Phrase (Fallback Text):", "I AM NOT TRASH").upper()
with col_opt2:
    mockup_choice = st.selectbox("Select 3D Merchandise Mockup:", ["Men's Classic Crew Neck T-Shirt", "Boutique Tote Bag"])

if st.button("🚀 Convert Sketch to Digital Font & Fill with Artwork", type="primary", use_container_width=True):
    if not art_file or not style_file:
        st.warning("Please upload BOTH Slot 1 (Artwork) and Slot 2 (Text Style) images.")
    else:
        with st.spinner("Extracting handwritten layout, converting to solid digital fonts, and clipping artwork..."):
            art_img = Image.open(art_file)
            art_img = ImageOps.exif_transpose(art_img)
            
            style_img = Image.open(style_file)
            style_img = ImageOps.exif_transpose(style_img)

            composited_result = composite_artwork_into_digital_typography(
                artwork_img=art_img,
                text_style_img=style_img,
                fallback_text=fallback_phrase
            )

            mockup_img = generate_3d_product_mockup(composited_result, mockup_choice)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("🖼️ Solid Digital Typography Filled with Artwork")
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
                    file_name="Solid_Digital_Typography_Artwork.png",
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
