import io
import os
import math
import zipfile
import urllib.request
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
    page_title="Commercial Typography & Google Fonts Studio",
    page_icon="🛍️",
    layout="wide"
)

MAX_SIZE = 1800
AUTO_STRAIGHTEN_DRAWING = True
MAX_STRAIGHTEN_ANGLE = 18

FONT_DIR = "./google_fonts_cache"
os.makedirs(FONT_DIR, exist_ok=True)

# ============================================================
# DYNAMIC GOOGLE FONTS ENGINE (50+ DISTINCT VISUAL STYLES)
# ============================================================

GOOGLE_FONTS_DB = {
    # CATEGORY 1: FLOWING CURSIVE & SCRIPT
    "Script - Pacifico": "https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf",
    "Script - Great Vibes": "https://github.com/google/fonts/raw/main/ofl/greatvibes/GreatVibes-Regular.ttf",
    "Script - Dancing Script": "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript-Bold.ttf",
    "Script - Sacramento": "https://github.com/google/fonts/raw/main/ofl/sacramento/Sacramento-Regular.ttf",
    "Script - Satisfy": "https://github.com/google/fonts/raw/main/ofl/satisfy/Satisfy-Regular.ttf",
    "Script - Kaushan Script": "https://github.com/google/fonts/raw/main/ofl/kaushanscript/KaushanScript-Regular.ttf",
    "Script - Caveat": "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat-Bold.ttf",

    # CATEGORY 2: HEAVY VINTAGE & DISPLAY BLOCK
    "Display - Bebas Neue": "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Display - Lobster": "https://github.com/google/fonts/raw/main/ofl/lobster/Lobster-Regular.ttf",
    "Display - Abril Fatface": "https://github.com/google/fonts/raw/main/ofl/abrilfatface/AbrilFatface-Regular.ttf",
    "Display - Black Ops One": "https://github.com/google/fonts/raw/main/ofl/blackopsone/BlackOpsOne-Regular.ttf",
    "Display - Bungee": "https://github.com/google/fonts/raw/main/ofl/bungee/Bungee-Regular.ttf",
    "Display - Alfa Slab One": "https://github.com/google/fonts/raw/main/ofl/alfaslabone/AlfaSlabOne-Regular.ttf",
    "Display - Righteous": "https://github.com/google/fonts/raw/main/ofl/righteous/Righteous-Regular.ttf",
    "Display - Anton": "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",

    # CATEGORY 3: PLAYFUL & HAND-DRAWN NURSERY
    "Playful - Amatic SC": "https://github.com/google/fonts/raw/main/ofl/amaticsc/AmaticSC-Bold.ttf",
    "Playful - Patrick Hand": "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "Playful - Indie Flower": "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "Playful - Shadows Into Light": "https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "Playful - Architects Daughter": "https://github.com/google/fonts/raw/main/ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf",
    "Playful - Luckiest Guy": "https://github.com/google/fonts/raw/main/ofl/luckiestguy/LuckiestGuy-Regular.ttf",
    "Playful - Fredoka": "https://github.com/google/fonts/raw/main/ofl/fredoka/Fredoka-Bold.ttf",
    "Playful - Sniglet": "https://github.com/google/fonts/raw/main/ofl/sniglet/Sniglet-ExtraBold.ttf",

    # CATEGORY 4: HIGH-FASHION LUXURY & SERIFS
    "Serif - Playfair Display": "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/PlayfairDisplay-Bold.ttf",
    "Serif - Bodoni Moda": "https://github.com/google/fonts/raw/main/ofl/bodonimoda/BodoniModa-Bold.ttf",
    "Serif - Cinzel": "https://github.com/google/fonts/raw/main/ofl/cinzel/Cinzel-Bold.ttf",
    "Serif - Cormorant Garamond": "https://github.com/google/fonts/raw/main/ofl/cormorantgaramond/CormorantGaramond-Bold.ttf",
    "Serif - Prata": "https://github.com/google/fonts/raw/main/ofl/prata/Prata-Regular.ttf",
    "Serif - DM Serif Display": "https://github.com/google/fonts/raw/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",

    # CATEGORY 5: ULTRA-MODERN & GEOMETRIC SANS
    "Modern - Montserrat": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-ExtraBold.ttf",
    "Modern - Oswald": "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf",
    "Modern - Syne": "https://github.com/google/fonts/raw/main/ofl/syne/Syne-Bold.ttf",
    "Modern - Outfit": "https://github.com/google/fonts/raw/main/ofl/outfit/Outfit-Bold.ttf",
    "Modern - Space Grotesk": "https://github.com/google/fonts/raw/main/ofl/spacegrotesk/SpaceGrotesk-Bold.ttf",
    "Modern - Comfortaa": "https://github.com/google/fonts/raw/main/ofl/comfortaa/Comfortaa-Bold.ttf"
}

def load_google_font(font_name: str, size: int):
    """Downloads Google Font TTF on demand and loads it directly into Pillow."""
    if font_name in GOOGLE_FONTS_DB:
        url = GOOGLE_FONTS_DB[font_name]
        safe_name = font_name.replace(" ", "_").replace("-", "_")
        filename = os.path.join(FONT_DIR, f"{safe_name}.ttf")
        
        if not os.path.exists(filename):
            try:
                urllib.request.urlretrieve(url, filename)
            except Exception:
                return ImageFont.load_default()
        try:
            return ImageFont.truetype(filename, size=size)
        except Exception:
            pass
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
# TYPOGRAPHY LAYOUT ENGINE (10 WRITING STYLES)
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

def generate_font_style_preview(phrase_text, font_obj):
    sample_text = phrase_text if phrase_text.strip() else "PREVIEW STYLE"
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), sample_text, font=font_obj)
    
    tw = max(200, bbox[2] - bbox[0] + 60)
    th = max(80, bbox[3] - bbox[1] + 40)
    
    preview_img = Image.new("RGBA", (tw, th), (250, 250, 252, 255))
    p_draw = ImageDraw.Draw(preview_img)
    p_draw.rectangle((0, 0, tw-1, th-1), outline=(200, 205, 215), width=2)
    tx = (tw - (bbox[2] - bbox[0])) // 2 - bbox[0]
    ty = (th - (bbox[3] - bbox[1])) // 2 - bbox[1]
    p_draw.text((tx, ty), sample_text, font=font_obj, fill=(20, 25, 35))
    return preview_img

def render_advanced_typography(phrase, font, font_label, drawings, mapping_mode, letter_configs, global_cfg, styling, writing_style):
    """Renders artwork across 10 distinct layout structural patterns."""
    rendered_letters = []
    letter_spacing = styling.get("letter_spacing", 15)
    stroke_expand = styling.get("stroke_expand", 8)

    # Calculate layout modifiers for writing styles
    if writing_style == "Interlocking / Merged Letter Tips":
        letter_spacing = -abs(letter_spacing) - 25
    elif writing_style == "Wide Modern Block Spacing":
        letter_spacing = max(letter_spacing, 45)

    clean_idx = 0
    char_list = list(phrase)

    for char_idx, char in enumerate(char_list):
        if char == " ":
            space_w = int(font.size * 0.4)
            rendered_letters.append({"is_space": True, "width": space_w, "height": 10})
            continue

        # Layout scaling for Drop Cap
        current_font = font
        if writing_style == "Drop Cap / Giant Initial Letter" and char_idx == 0:
            current_font = load_google_font(font_label, int(font.size * 2.2))

        mask, lw, lh = render_letter_mask(char, current_font, stroke_expand=stroke_expand)
        
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

        # Apply Outline
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

        # Apply Shadow
        if styling.get("enable_shadow"):
            shadow_mask = mask.filter(ImageFilter.GaussianBlur(styling.get("shadow_blur", 10)))
            shadow_tile = Image.new("RGBA", (lw + 20, lh + 20), (0, 0, 0, 0))
            shadow_img = Image.new("RGBA", (lw, lh), (0, 0, 0, 140))
            shadow_tile.paste(shadow_img, (10, 10), shadow_mask)
            
            combined = Image.new("RGBA", (lw + 20, lh + 20), (0, 0, 0, 0))
            combined.alpha_composite(shadow_tile)
            combined.alpha_composite(letter_tile, (0, 0))
            letter_tile = combined
            lw, lh = lw + 20, lh + 20

        rendered_letters.append({"is_space": False, "image": letter_tile, "width": lw, "height": lh, "char": char})

    # Assemble Layout Base Dimensions
    total_w = sum(item["width"] for item in rendered_letters) + (len(rendered_letters) * letter_spacing)
    max_h = max((item["height"] for item in rendered_letters if not item["is_space"]), default=100)
    
    canvas_w = total_w + 200
    canvas_h = max_h + 300
    bg_color = styling.get("bg_color", "#FFFFFF")

    if bg_color == "TRANSPARENT":
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    else:
        bg_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_rgb + (255,))

    curr_x = 100
    num_chars = max(1, len(rendered_letters))

    for idx, item in enumerate(rendered_letters):
        if item["is_space"]:
            curr_x += item["width"] + letter_spacing
            continue

        curr_y = (canvas_h - item["height"]) // 2

        # 10 Writing Style Vertical & Trajectory Alterations
        if writing_style == "Staggered / Bounce Baseline":
            curr_y += -35 if (idx % 2 == 0) else 35
        elif writing_style == "Wavy / Sine-Wave Flow":
            curr_y += int(math.sin((idx / float(num_chars)) * math.pi * 2) * 45)
        elif writing_style == "Arched / Upward Curve Path":
            mid = num_chars / 2.0
            curr_y -= int((1 - ((idx - mid) / mid) ** 2) * 60) if mid > 0 else 0
        elif writing_style == "Trapezoid / Envelope Varsity Warp":
            mid = num_chars / 2.0
            scale = 1.0 + (1 - abs(idx - mid) / mid) * 0.35 if mid > 0 else 1.0
            item_img = item["image"].resize((int(item["width"] * scale), int(item["height"] * scale)), Image.Resampling.LANCZOS)
            canvas.alpha_composite(item_img, (curr_x, curr_y - (item_img.height - item["height"]) // 2))
            curr_x += item_img.width + letter_spacing
            continue

        canvas.alpha_composite(item["image"], (curr_x, curr_y))
        curr_x += item["width"] + letter_spacing

    return canvas, rendered_letters

def generate_3d_product_mockup(artwork: Image.Image, apparel_style: str) -> Image.Image:
    mockup = Image.new("RGBA", (1200, 1200), (238, 240, 245, 255))
    draw = ImageDraw.Draw(mockup)

    if apparel_style == "Men's Classic Crew Neck T-Shirt":
        shirt_pts = [(350, 140), (450, 90), (750, 90), (850, 140), (1050, 310), (940, 460), (870, 410), (870, 1120), (330, 1120), (330, 410), (260, 460), (150, 310)]
        draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(190, 190, 190), width=4)
        draw.arc((500, 70, 700, 170), start=0, end=180, fill=(180, 180, 180), width=5)
        draw.line((330, 410, 330, 1120), fill=(210, 210, 210, 120), width=15)
        draw.line((870, 410, 870, 1120), fill=(210, 210, 210, 120), width=15)
        
        target_w = 460
        scale = target_w / float(artwork.width)
        target_h = int(artwork.height * scale)
        art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (370, 380))

    elif apparel_style == "Women's Fitted Curve T-Shirt":
        shirt_pts = [(380, 140), (460, 90), (740, 90), (820, 140), (1000, 300), (910, 420), (830, 370), (800, 700), (840, 1100), (360, 1100), (400, 700), (370, 370), (290, 420), (200, 300)]
        draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(190, 190, 190), width=4)
        draw.arc((480, 70, 720, 210), start=0, end=180, fill=(180, 180, 180), width=4)
        
        target_w = 380
        scale = target_w / float(artwork.width)
        target_h = int(artwork.height * scale)
        art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (410, 380))

    elif apparel_style == "Sleeveless Tank Top / Vest":
        shirt_pts = [(400, 120), (470, 80), (730, 80), (800, 120), (820, 360), (840, 1120), (360, 1120), (380, 360)]
        draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(190, 190, 190), width=4)
        draw.arc((470, 60, 730, 250), start=0, end=180, fill=(180, 180, 180), width=5)
        
        target_w = 400
        scale = target_w / float(artwork.width)
        target_h = int(artwork.height * scale)
        art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (400, 360))

    elif apparel_style == "V-Neck T-Shirt":
        shirt_pts = [(350, 140), (450, 90), (750, 90), (850, 140), (1050, 310), (940, 460), (870, 410), (870, 1120), (330, 1120), (330, 410), (260, 460), (150, 310)]
        draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(190, 190, 190), width=4)
        draw.polygon([(480, 90), (600, 280), (720, 90)], outline=(180, 180, 180), fill=(238, 240, 245), width=4)
        
        target_w = 440
        scale = target_w / float(artwork.width)
        target_h = int(artwork.height * scale)
        art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (380, 410))

    elif apparel_style == "Long Sleeve Sweatshirt":
        shirt_pts = [(350, 140), (450, 90), (750, 90), (850, 140), (1150, 650), (1020, 750), (870, 440), (870, 1120), (330, 1120), (330, 440), (180, 750), (50, 650)]
        draw.polygon(shirt_pts, fill=(255, 255, 255), outline=(190, 190, 190), width=4)
        draw.arc((500, 70, 700, 170), start=0, end=180, fill=(180, 180, 180), width=6)
        
        target_w = 460
        scale = target_w / float(artwork.width)
        target_h = int(artwork.height * scale)
        art_resized = artwork.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (370, 400))

    else: # Tote Bag
        draw.line((400, 80, 400, 380), fill=(180, 160, 130), width=32)
        draw.line((800, 80, 800, 380), fill=(180, 160, 130), width=32)
        draw.polygon([(240, 380), (960, 380), (910, 1120), (290, 1120)], fill=(248, 244, 230), outline=(190, 180, 160), width=5)
        
        target_w = 500
        scale = min(500 / artwork.width, 500 / artwork.height)
        new_w, new_h = int(artwork.width * scale), int(artwork.height * scale)
        art_resized = artwork.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mockup.alpha_composite(art_resized, (240 + (720 - new_w) // 2, 380 + (740 - new_h) // 2))

    return mockup

# ============================================================
# STREAMLIT APPLICATION STATE & WORKFLOW
# ============================================================

if "drawings" not in st.session_state:
    st.session_state["drawings"] = []

st.title("🛍️ Commercial Typography & Multi-Font Studio")
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

    # --- STEP 2: PHRASE & GOOGLE FONT GENERATION MODE ---
    st.markdown("---")
    st.header("Step 2: Choose Phrase & Google Fonts Generation Mode")

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        phrase = st.text_input("Enter Text / Phrase:", "CREATIVE").upper()
    with col_t2:
        font_generation_mode = st.radio(
            "Font Generation Selection:",
            ["Single Specific Font", "Batch Render Multiple Fonts At Once"],
            horizontal=True
        )

    all_font_labels = list(GOOGLE_FONTS_DB.keys())

    if font_generation_mode == "Single Specific Font":
        selected_fonts = [st.selectbox("Select Primary Google Font Style:", all_font_labels)]
    else:
        num_fonts_option = st.selectbox(
            "How many distinct font variations to generate?",
            ["Top 5 Fonts", "Top 10 Fonts", "Top 20 Fonts", "ALL Available Fonts (30+)"]
        )
        if num_fonts_option == "Top 5 Fonts":
            selected_fonts = all_font_labels[:5]
        elif num_fonts_option == "Top 10 Fonts":
            selected_fonts = all_font_labels[:10]
        elif num_fonts_option == "Top 20 Fonts":
            selected_fonts = all_font_labels[:20]
        else:
            selected_fonts = all_font_labels

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        font_size = st.slider("Font Base Size (px):", 120, 500, 220)
    with col_s2:
        stroke_expand = st.slider("Extra Letter Thickness:", 0, 30, 8)
    with col_s3:
        letter_spacing = st.slider("Letter Spacing (Gap):", -20, 100, 15)

    primary_font_label = selected_fonts[0]
    primary_font = load_google_font(primary_font_label, font_size)

    # Live Font Style Preview
    st.subheader("🔤 Live Google Font Style Preview")
    font_preview_img = generate_font_style_preview(phrase, primary_font)
    st.image(font_preview_img, caption=f"Primary Font: {primary_font_label}", use_container_width=True)

    # --- STEP 3: WRITING STYLES & ARTWORK MAPPING ---
    st.markdown("---")
    st.header("Step 3: Writing Style Structural Layout & Artwork Mapping")

    writing_style = st.selectbox(
        "Select Typographic Writing / Structure Layout Pattern (10 Options):",
        [
            "Standard Horizontal Alignment",
            "Drop Cap / Giant Initial Letter",
            "Interlocking / Merged Letter Tips",
            "Arched / Upward Curve Path",
            "Staggered / Bounce Baseline",
            "Wavy / Sine-Wave Flow",
            "Trapezoid / Envelope Varsity Warp",
            "Wide Modern Block Spacing"
        ]
    )

    mapping_mode = st.radio(
        "Mapping Options:",
        ["Per-Letter Assignment (Choose drawing & crop region for EACH letter)",
         "Entire Phrase (Single continuous drawing spans across whole text)"],
        horizontal=True
    )

    clean_phrase = [c for c in phrase if c.strip()]
    letter_configs = {}
    global_cfg = {"drawing_idx": 0, "crop_x": 50, "crop_y": 50}

    if mapping_mode.startswith("Per-Letter") and clean_phrase:
        st.subheader("Fine-Tune Individual Letters & Artwork Previews")
        for idx, char in enumerate(clean_phrase):
            with st.expander(f"Letter #{idx + 1}: '{char}' Configuration & Preview", expanded=(idx == 0)):
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

                # Live Individual Letter Artwork Preview
                mask, lw, lh = render_letter_mask(char, primary_font, stroke_expand=stroke_expand)
                src_img = st.session_state["drawings"][assigned_drawing_idx]["image"]
                texture_portion = process_image_portion(src_img, crop_x, crop_y, zoom)
                texture_resized = texture_portion.resize((lw, lh), Image.Resampling.LANCZOS).convert("RGBA")
                letter_preview = Image.new("RGBA", (lw, lh), (255, 255, 255, 255))
                letter_preview.paste(texture_resized, (0, 0), mask)
                st.image(letter_preview, caption=f"Live Artwork Preview for Letter '{char}'", width=140)

    else:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            global_cfg["drawing_idx"] = st.selectbox(
                "Drawing to span across whole phrase:",
                options=range(len(st.session_state["drawings"])),
                format_func=lambda x: f"Drawing #{x + 1}"
            )
        with sc2:
            global_cfg["crop_x"] = st.slider("Horizontal Texture Shift %:", 0, 100, 50)
        with sc3:
            global_cfg["crop_y"] = st.slider("Vertical Texture Shift %:", 0, 100, 50)

    # Style Controls
    st.subheader("Professional Styling & 3D Mockup Model")
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
        mockup_choice = st.selectbox(
            "Select 3D Merchandise Mockup Pattern:",
            [
                "Men's Classic Crew Neck T-Shirt",
                "Women's Fitted Curve T-Shirt",
                "Sleeveless Tank Top / Vest",
                "V-Neck T-Shirt",
                "Long Sleeve Sweatshirt",
                "Boutique Tote Bag"
            ]
        )

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
    st.header(f"Step 4: Render Artwork ({len(selected_fonts)} Google Font(s) Selected)")
    
    button_label = f"🚀 Render Artwork for {len(selected_fonts)} Font(s) ({writing_style}) & 3D Mockup"
    if st.button(button_label, type="primary", use_container_width=True):
        if not phrase.strip():
            st.warning("Please enter a text phrase.")
        else:
            with st.spinner(f"Downloading Google Fonts and rendering '{writing_style}' artwork..."):
                rendered_font_results = {}
                primary_canvas = None
                primary_letters = None

                for font_label in selected_fonts:
                    f_obj = load_google_font(font_label, font_size)
                    canvas, letters = render_advanced_typography(
                        phrase=phrase,
                        font=f_obj,
                        font_label=font_label,
                        drawings=st.session_state["drawings"],
                        mapping_mode=mapping_mode,
                        letter_configs=letter_configs,
                        global_cfg=global_cfg,
                        styling=styling_opts,
                        writing_style=writing_style
                    )
                    rendered_font_results[font_label] = canvas
                    if primary_canvas is None:
                        primary_canvas = canvas
                        primary_letters = letters

                # Generate 3D Mockup
                mockup_img = generate_3d_product_mockup(primary_canvas, mockup_choice)

                # Display Results
                if len(selected_fonts) == 1:
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.subheader(f"🖼️ High-Res Output ({writing_style})")
                        st.image(primary_canvas, use_container_width=True)
                    with res_col2:
                        st.subheader(f"👕 Live 3D {mockup_choice} Mockup")
                        st.image(mockup_img, use_container_width=True)
                else:
                    st.subheader(f"🖼️ All-Font Output Gallery ({len(rendered_font_results)} Styles)")
                    grid_cols = st.columns(min(3, len(rendered_font_results)))
                    for idx, (f_name, f_canvas) in enumerate(rendered_font_results.items()):
                        with grid_cols[idx % len(grid_cols)]:
                            st.markdown(f"**Font #{idx + 1}: {f_name}**")
                            st.image(f_canvas, use_container_width=True)

                    st.subheader(f"👕 Primary Style ({selected_fonts[0]}) on 3D Mockup")
                    st.image(mockup_img, use_container_width=True)

                # Multi-Font & Letter ZIP Exporters
                font_zip_buf = io.BytesIO()
                with zipfile.ZipFile(font_zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for f_name, f_canvas in rendered_font_results.items():
                        b = io.BytesIO()
                        f_canvas.save(b, format="PNG", dpi=(300, 300))
                        clean_fname = f_name.replace(" ", "_").replace("/", "_")
                        zip_file.writestr(f"Fonts/{clean_fname}.png", b.getvalue())

                letter_zip_buf = io.BytesIO()
                with zipfile.ZipFile(letter_zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, item in enumerate(primary_letters):
                        if not item["is_space"]:
                            img_buf = io.BytesIO()
                            item["image"].save(img_buf, format="PNG", dpi=(300, 300))
                            char_name = item.get("char", f"Letter_{idx+1}")
                            zip_file.writestr(f"Letters/{idx+1}_{char_name}_Transparent.png", img_buf.getvalue())

                # Download Buttons
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                with dl_col1:
                    buf1 = io.BytesIO()
                    primary_canvas.save(buf1, format="PNG", dpi=(300, 300))
                    st.download_button(
                        label="📥 Download Primary Artwork (PNG)",
                        data=buf1.getvalue(),
                        file_name=f"{phrase}_Primary_Artwork.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with dl_col2:
                    st.download_button(
                        label=f"📦 Download All {len(selected_fonts)} Font Variations (ZIP)",
                        data=font_zip_buf.getvalue(),
                        file_name=f"{phrase}_All_Font_Variations.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                with dl_col3:
                    st.download_button(
                        label="📦 Download Individual Letters (ZIP Asset Pack)",
                        data=letter_zip_buf.getvalue(),
                        file_name=f"{phrase}_Individual_Letters_Pack.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
