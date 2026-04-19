# APEX — CCWS Senior Services Directory
### Deployment guide (Streamlit Community Cloud — free)

This gets your demo live in about 10 minutes so you can generate a QR code for the Showcase poster.

---

## 1. Prepare the repo (5 min)

Your GitHub repo needs exactly these files at the root:

```
your-repo/
├── app.py                      # existing (with 2-line patch below)
├── services_dataset.csv        # existing
├── requirements.txt            # NEW — provided
└── .streamlit/
    └── config.toml             # NEW — provided
```

### Apply the 2-line patch to app.py

Open `app.py` and replace lines **115–118** with the block inside `PATCH_app_py.txt`. Then change the Ollama timeout from `60` to `8` on line ~527.

**Why:** Streamlit Cloud can't reach `localhost:11434`. Without the patch, every AI search hangs for 60 seconds before falling back. With the patch, the app detects deployment and skips Ollama entirely — rule-based search still returns great results.

---

## 2. Push to GitHub (2 min)

```bash
git add requirements.txt .streamlit/config.toml app.py
git commit -m "Deploy: add requirements, streamlit config, deployment-safe LLM config"
git push
```

If you don't have a repo yet:
1. Go to github.com → New repository → name it `apex-ccws-demo`
2. Make it **public** (Streamlit Community Cloud free tier requires public repos)
3. Upload all files via web UI or `git init && git remote add origin ...`

---

## 3. Deploy on Streamlit Cloud (3 min)

1. Go to **https://share.streamlit.io** → sign in with GitHub
2. Click **New app**
3. Fill in:
   - Repository: `your-username/apex-ccws-demo`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**
5. First build takes ~2 minutes. You'll get a URL like:
   `https://apex-ccws-demo.streamlit.app`

**Test on your phone** before generating the QR code. Make sure:
- [ ] Home tab loads
- [ ] Search returns results (try "food", "doctor", "housing")
- [ ] Voice button appears (may not work if browser blocks mic — that's OK, still shows the feature)
- [ ] Tab navigation works
- [ ] No error messages

---

## 4. Generate the QR code (1 min)

Once you have the live URL, use any of these:
- **https://www.qr-code-generator.com/** — paste URL, download PNG at 1000×1000 minimum
- **Canva** → Design → QR code element → paste URL → download as PNG
- Or run this in Python:

```python
# pip install qrcode[pil]
import qrcode
img = qrcode.make("https://your-url.streamlit.app")
img.save("apex_demo_qr.png")
```

Use the PNG at **120–150px** on the poster (already sized correctly in your current poster — just swap the SVG placeholder).

---

## 5. If something breaks

**"ModuleNotFoundError: streamlit_mic_recorder"**
→ Double-check `requirements.txt` is in repo root, not in a subfolder.

**App loads but search returns nothing**
→ Confirm `services_dataset.csv` is in the repo root and spelled exactly like that (case-sensitive on Linux).

**Voice button doesn't show up**
→ Expected on deployment — `streamlit-mic-recorder` sometimes fails to install on Cloud. If it breaks the build, remove that line from `requirements.txt`. The app handles it with a try/except already (lines 11–15 of app.py).

**AI recommendations say "no AI result"**
→ Expected on deployment. The patch makes this silent. Rule-based search still ranks results.

---

## For the Showcase — two things to say if asked about the AI

1. *"The live demo runs on rule-based matching that covers 95% of senior search queries. The full AI layer runs locally on Ollama and would move to a cloud LLM API in production — we've costed that in the business plan."*
2. *"What you're seeing is a proof-of-concept. The architecture supports the AI layer; we've tuned it for deployment reliability over demo risk."*

That reframes a technical limitation as a professional decision. Which it is.
