import io
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# Page configuration
st.set_page_config(
    page_title="Dual-Slot Custom Lettering Studio",
    page_icon="🎨",
    layout="wide"
)

MAX_SIZE = 1800

# ============================================================
# SHADOW-IMMUNE EXTRACTION ENGINE (Extracts Hand-Drawn Mask)
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

def extract_handwritten_mask(img_rgb):
    """Extracts black ink strokes from paper into an alpha mask."""
    arr = np.array(img_rgb).astype(np.uint8)
    h, w = arr.shape[:2]

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Gaussian thresholding to isolate ink on paper
    bg_size = max(61, int(min(h, w) * 0.10))
    if bg_size % 2 == 0:
        bg_size += 1

    bg_gray = cv2.GaussianBlur(gray, (bg_size, bg_size), 0)
    local_darkness = bg_gray.astype(np.float32) - gray.astype(np.float32)
    
    # Stroke mask
    mask = np.where(local_darkness > 15.0, 255, 0).astype(np.uint8)
    
    # Additional edge detection for thin strokes
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_mask = np.where(magnitude > 25.0, 255, 0).astype(np.uint8)

    combined_mask = cv2.bitwise_or(mask, edge_mask)
    
    # Morphological cleaning
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned_mask = remove_small_components(cleaned_mask, min_area=25)
    
    return cleaned_mask

# ============================================================
# COMPOSITING ENGINE: ARTWORK FILLED INTO TEXT STROKES
# ============================================================

def composite_artwork_into_text_style(artwork_img, text_style_img, stroke_expansion=0):
    """Clips Upload #1 (Artwork Pattern) inside the strokes of Upload #2 (Text Style)."""
    
    art_rgb = artwork_img.convert("RGB")
    style_rgb = text_style_img.convert("RGB")
    
    # Match resolution to style image
    style_rgb = resize_image(style_rgb, MAX_SIZE)
    w, h = style_rgb.size
    art_rgb = art_rgb.resize((w, h), Image.Resampling.LANCZOS)
    
    # Extract mask of hand-written text strokes
    stroke_mask = extract_handwritten_mask(style_rgb)
    
    if stroke_expansion > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_expansion * 2 + 1, stroke_expansion * 2 + 1))
        stroke_mask = cv2.dilate(stroke_mask, k, iterations=1)
        
    stroke_mask = cv2.GaussianBlur(stroke_mask, (3, 3), 0)
    
    # Clip Artwork into stroke mask
    art_arr = np.array(art_rgb)
    rgba_arr = np.dstack((art_arr, stroke_mask))
    composited = Image.fromarray(rgba_arr, "RGBA")
    
    # Crop to bounding box of content
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

st.title("🎨 Dual-Slot Custom Lettering & Artwork Studio")
st.write("Upload artwork texture in **Slot 1** and hand-drawn lettering style in **Slot 2**. The app clips the artwork directly inside the strokes!")

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
    st.header("Slot 2: Text Style & Layout")
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
st.header("Step 3: Render Composited Artwork & 3D Merch Preview")

col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    stroke_expand = st.slider("Thicken Letter Strokes:", 0, 10, 2)
with col_opt2:
    mockup_choice = st.selectbox("Select 3D Merchandise Mockup:", ["Men's Classic Crew Neck T-Shirt", "Boutique Tote Bag"])

if st.button("🚀 Fill Artwork into Handwritten Text Style", type="primary", use_container_width=True):
    if not art_file or not style_file:
        st.warning("Please upload BOTH Slot 1 (Artwork) and Slot 2 (Text Style) images.")
    else:
        with st.spinner("Processing mask extraction and clipping artwork pattern..."):
            art_img = Image.open(art_file)
            art_img = ImageOps.exif_transpose(art_img)
            
            style_img = Image.open(style_file)
            style_img = ImageOps.exif_transpose(style_img)

            # Core compositing step
            composited_result = composite_artwork_into_text_style(
                artwork_img=art_img,
                text_style_img=style_img,
                stroke_expansion=stroke_expand
            )

            mockup_img = generate_3d_product_mockup(composited_result, mockup_choice)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("🖼️ Artwork Filled into Hand-Drawn Text Style")
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
                    file_name="Artwork_Filled_Lettering.png",
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
