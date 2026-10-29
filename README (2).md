# AniMuse AI — Streamlit deployment

A Streamlit version of AniMuse AI using the pretrained AnimeGANv2 Hayao ONNX model.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app downloads `AnimeGANv2_Hayao.onnx` from Hugging Face on first startup and stores it under `models/`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository, for example `animuse-ai-streamlit`.
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the repository root. Do not upload the model file; the app downloads it at startup.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Click **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Deploy. The first start can take longer because dependencies and the model are downloaded.

## Notes

- CPU inference is used for compatibility.
- Streamlit Community Cloud resources are limited; very large images may process slowly.
- Generation history is stored in Streamlit session state and is temporary.
- Background Studio makes a gradient poster composition. It does not remove the original photo background or generate realistic scenery.
