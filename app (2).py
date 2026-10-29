import io
import os
import uuid
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import onnxruntime as ort
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont, ImageFilter

APP_TITLE = "AniMuse AI"
MODEL_URL = "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "AnimeGANv2_Hayao.onnx"
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)

MOODS = ["Classic Anime", "Dreamy Pastel", "Sunset Glow", "Midnight Blue", "Vibrant Anime", "Soft Vintage"]
THEMES = ["Auto Scene", "Sakura Pink", "Sunset", "Ocean", "Midnight", "Fantasy", "Mint"]
EXPORT_FORMATS = {
    "Phone Wallpaper (1080 × 1920)": (1080, 1920),
    "Desktop Wallpaper (1920 × 1080)": (1920, 1080),
    "Square Avatar (1024 × 1024)": (1024, 1024),
    "Profile Avatar (512 × 512)": (512, 512),
}

st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide")
st.markdown("""
<style>
.stApp { background: #E8E6E1; color: #302B2D; }
[data-testid="stHeader"] { background: transparent; }
.hero { background: linear-gradient(125deg,#541F35,#71364D,#7F8F7C); padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem; }
.hero h1, .hero p { color: #FAF8F5 !important; margin: 0; }
.stButton > button, .stDownloadButton > button { background: #541F35; color: #FAF8F5; border: 1px solid #541F35; border-radius: 10px; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #71364D; color: white; border-color: #71364D; }
</style>
<div class="hero"><h1>AniMuse AI</h1><p>Transform your photos into anime-inspired art.</p></div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AnimeGANv2 model…")
def load_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        response = requests.get(MODEL_URL, timeout=300)
        response.raise_for_status()
        temp_path = MODEL_PATH.with_suffix(".download")
        temp_path.write_bytes(response.content)
        if temp_path.stat().st_size < 1_000_000:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError("The downloaded model file looks incomplete.")
        temp_path.replace(MODEL_PATH)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0], session.get_outputs()[0]

def get_model():
    try:
        return load_model()
    except Exception as exc:
        st.error(f"Could not load AnimeGANv2 model: {exc}")
        st.info("Check that the app can access Hugging Face and try refreshing the page.")
        st.stop()

session, model_input, model_output = get_model()

def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def remember(image: Image.Image, label: str):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {"label": label, "image": image.copy(), "id": uuid.uuid4().hex[:8]})
    st.session_state.history = st.session_state.history[:30]

def stylize(image: Image.Image, mood: str = "Classic Anime", strength: float = 1.0) -> Image.Image:
    if image is None:
        raise ValueError("Please upload a photo first.")
    original = ImageOps.exif_transpose(image).convert("RGB")
    resized = original.resize((512, 512), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    arr = arr[None, ...]
    output = session.run([model_output.name], {model_input.name: arr})[0]
    output = np.asarray(output)
    if output.ndim == 4:
        output = output[0]
    if output.ndim == 3 and output.shape[-1] != 3:
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected model output shape: {output.shape}")
    if output.ndim != 3 or output.shape[-1] != 3:
        raise ValueError(f"Unexpected model output shape: {output.shape}")
    if output.min() < 0:
        output = (output + 1.0) * 127.5
    elif output.max() <= 1.0:
        output = output * 255.0
    output = np.clip(output, 0, 255).astype(np.uint8)
    anime = Image.fromarray(output).resize(original.size, Image.Resampling.LANCZOS)
    if mood == "Dreamy Pastel":
        anime = ImageEnhance.Color(anime).enhance(0.75)
        anime = ImageEnhance.Brightness(anime).enhance(1.08)
        anime = ImageEnhance.Contrast(anime).enhance(0.90)
    elif mood == "Sunset Glow":
        a = np.asarray(anime).astype(np.float32)
        a[:, :, 0] *= 1.10; a[:, :, 1] *= 1.02; a[:, :, 2] *= 0.86
        anime = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif mood == "Midnight Blue":
        a = np.asarray(anime).astype(np.float32)
        a[:, :, 0] *= 0.84; a[:, :, 1] *= 0.94; a[:, :, 2] *= 1.13
        anime = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        anime = ImageEnhance.Contrast(anime).enhance(1.08)
    elif mood == "Vibrant Anime":
        anime = ImageEnhance.Color(anime).enhance(1.35)
        anime = ImageEnhance.Contrast(anime).enhance(1.08)
    elif mood == "Soft Vintage":
        anime = ImageEnhance.Color(anime).enhance(0.78)
        anime = ImageEnhance.Contrast(anime).enhance(0.90)
        anime = ImageEnhance.Brightness(anime).enhance(1.04)
    return Image.blend(original, anime, float(np.clip(strength, 0.0, 1.0)))

def make_poster(image: Image.Image, theme: str, mood: str) -> Image.Image:
    palettes = {
        "Auto Scene": ((225, 230, 218), (83, 31, 53)),
        "Sakura Pink": ((255, 222, 232), (132, 75, 111)),
        "Sunset": ((255, 207, 154), (128, 61, 81)),
        "Ocean": ((191, 229, 231), (36, 82, 117)),
        "Midnight": ((88, 91, 137), (22, 25, 54)),
        "Fantasy": ((226, 211, 246), (86, 68, 129)),
        "Mint": ((218, 238, 221), (53, 103, 91)),
    }
    width, height = 900, 1200
    top, bottom = palettes.get(theme, palettes["Auto Scene"])
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / (height - 1)
        arr[y, :, :] = [int(top[c] * (1 - t) + bottom[c] * t) for c in range(3)]
    background = Image.fromarray(arr, "RGB").convert("RGBA")
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((180, 100, 720, 700), fill=(255, 244, 220, 100))
    glow = glow.filter(ImageFilter.GaussianBlur(65))
    background = Image.alpha_composite(background, glow).convert("RGB")
    anime = ImageOps.fit(stylize(image, mood, 1.0), (width, height), method=Image.Resampling.LANCZOS)
    poster = Image.blend(background, anime, 0.68)
    draw = ImageDraw.Draw(poster)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        title_font = small_font = ImageFont.load_default()
    draw.text((35, 25), "ANIMUSE AI", fill="white", font=title_font)
    draw.text((37, 70), f"{mood.upper()} | PORTRAIT EDITION", fill="white", font=small_font)
    return poster

def make_comic(image: Image.Image, mood: str, dialogue: str) -> Image.Image:
    anime = stylize(image, mood, 1.0)
    page = Image.new("RGB", (900, 1100), "#f6f1ff")
    draw = ImageDraw.Draw(page)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        title_font = text_font = ImageFont.load_default()
    draw.text((30, 20), "MY ANIME STORY", fill="#33264d", font=title_font)
    positions = [(30, 90), (470, 90), (250, 520)]
    captions = ["A new adventure begins...", (dialogue or "I will never give up!")[:38], "To be continued..."]
    for i, (x, y) in enumerate(positions):
        panel = ImageOps.fit(anime, (400, 380), method=Image.Resampling.LANCZOS, centering=(0.5, 0.4))
        page.paste(panel, (x, y))
        draw.rectangle((x, y, x + 400, y + 380), outline="#201a30", width=5)
        draw.rounded_rectangle((x + 12, y + 12, x + 388, y + 68), radius=12, fill="white", outline="#201a30", width=2)
        draw.text((x + 20, y + 30), captions[i], fill="#201a30", font=text_font)
    draw.text((30, 1030), "Created with AniMuse AI", fill="#33264d", font=text_font)
    return page

st.caption("Choose a tab to create anime artwork, posters, comics, wallpapers, or avatars.")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Anime Generator", "Background Studio", "Manga Comic", "Wallpaper & Avatar", "Generation History"])

with tab1:
    left, right = st.columns(2)
    with left:
        upload = st.file_uploader("Upload your photo", type=["png", "jpg", "jpeg", "webp"], key="anime_upload")
        mood = st.selectbox("Anime mood", MOODS, key="anime_mood")
        strength = st.slider("Transformation strength", 0.2, 1.0, 1.0, 0.1, key="anime_strength")
        if st.button("Generate Anime", type="primary", key="generate_anime"):
            if upload is None:
                st.warning("Please upload a photo first.")
            else:
                try:
                    with st.spinner("Generating anime image…"):
                        source = Image.open(upload).convert("RGB")
                        result = stylize(source, mood, strength)
                        st.session_state.anime_result = result
                        remember(result, "Anime portrait")
                except Exception as exc:
                    st.error(f"Image generation failed: {exc}")
    with right:
        if "anime_result" in st.session_state:
            st.image(st.session_state.anime_result, caption="Anime result", use_container_width=True)
            st.download_button("Download PNG", image_to_bytes(st.session_state.anime_result), "animuse_anime.png", "image/png", key="download_anime")

with tab2:
    left, right = st.columns(2)
    with left:
        poster_upload = st.file_uploader("Upload your photo", type=["png", "jpg", "jpeg", "webp"], key="poster_upload")
        theme = st.selectbox("Scene style", THEMES, key="poster_theme")
        poster_mood = st.selectbox("Anime mood", MOODS, key="poster_mood")
        if st.button("Create Poster", type="primary", key="create_poster"):
            if poster_upload is None:
                st.warning("Please upload a photo first.")
            else:
                try:
                    with st.spinner("Creating poster…"):
                        result = make_poster(Image.open(poster_upload).convert("RGB"), theme, poster_mood)
                        st.session_state.poster_result = result
                        remember(result, "Poster")
                except Exception as exc:
                    st.error(f"Poster creation failed: {exc}")
    with right:
        if "poster_result" in st.session_state:
            st.image(st.session_state.poster_result, caption="Poster", use_container_width=True)
            st.download_button("Download poster", image_to_bytes(st.session_state.poster_result), "animuse_poster.png", "image/png", key="download_poster")
    st.info("This version creates a gradient poster composition. It does not automatically remove the original photo background or generate a new realistic scene.")

with tab3:
    left, right = st.columns(2)
    with left:
        comic_upload = st.file_uploader("Upload your photo", type=["png", "jpg", "jpeg", "webp"], key="comic_upload")
        comic_mood = st.selectbox("Anime mood", MOODS, key="comic_mood")
        dialogue = st.text_input("Main speech bubble", "I will never give up!", key="comic_dialogue")
        if st.button("Create Comic Page", type="primary", key="create_comic"):
            if comic_upload is None:
                st.warning("Please upload a photo first.")
            else:
                try:
                    with st.spinner("Creating manga page…"):
                        result = make_comic(Image.open(comic_upload).convert("RGB"), comic_mood, dialogue)
                        st.session_state.comic_result = result
                        remember(result, "Manga comic")
                except Exception as exc:
                    st.error(f"Comic creation failed: {exc}")
    with right:
        if "comic_result" in st.session_state:
            st.image(st.session_state.comic_result, caption="Manga page", use_container_width=True)
            st.download_button("Download comic", image_to_bytes(st.session_state.comic_result), "animuse_comic.png", "image/png", key="download_comic")

with tab4:
    left, right = st.columns(2)
    with left:
        export_upload = st.file_uploader("Upload your photo", type=["png", "jpg", "jpeg", "webp"], key="export_upload")
        export_mood = st.selectbox("Anime mood", MOODS, key="export_mood")
        export_format = st.selectbox("Export format", list(EXPORT_FORMATS.keys()), key="export_format")
        if st.button("Create Export", type="primary", key="create_export"):
            if export_upload is None:
                st.warning("Please upload a photo first.")
            else:
                try:
                    with st.spinner("Preparing export…"):
                        source = Image.open(export_upload).convert("RGB")
                        size = EXPORT_FORMATS[export_format]
                        result = ImageOps.fit(stylize(source, export_mood, 1.0), size, method=Image.Resampling.LANCZOS)
                        st.session_state.export_result = result
                        remember(result, export_format)
                except Exception as exc:
                    st.error(f"Export creation failed: {exc}")
    with right:
        if "export_result" in st.session_state:
            st.image(st.session_state.export_result, caption="Export preview", use_container_width=True)
            st.download_button("Download export", image_to_bytes(st.session_state.export_result), "animuse_export.png", "image/png", key="download_export")

with tab5:
    history = st.session_state.get("history", [])
    if not history:
        st.info("No creations in this session yet. Generate an image first.")
    else:
        for item in history:
            with st.expander(f"{item['label']} — {item['id']}"):
                st.image(item["image"], use_container_width=True)
                st.download_button("Download this image", image_to_bytes(item["image"]), f"animuse_{item['id']}.png", "image/png", key=f"history_{item['id']}")

st.markdown("---")
st.caption("AniMuse AI • Anime stylization powered by AnimeGANv2 and ONNX Runtime")
