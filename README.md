# APEX × CCWS — Senior Services Directory

**Live demo:** [apex-ccws.streamlit.app](https://apex-ccws.streamlit.app)

A plain-language, AI-powered directory built for seniors in Greater Victoria. One search box. Plain words. No login. Designed so seniors don't need to be a detective to find help that already exists in their neighborhood.

## The problem

Hundreds of senior services exist across Greater Victoria, but information is fragmented across 8+ separate websites — each with different jargon, layouts, and update schedules. The average senior spends 47 minutes searching, and 1 in 3 give up before finding what they need.

## What we built

A senior-first directory with:

- **Plain-language search** — Ask the way you'd actually talk: "I need help getting food delivered."
- **AI-powered matching** — Llama-based smart search understands intent, not just keywords.
- **Voice search** — Tap to talk for users who find typing difficult.
- **Saved favorites** — Keep services you've found.
- **Senior-first design** — Larger text, clear contrast, simple navigation.
- **40+ verified services** — Covering food, transport, housing, and health across Greater Victoria.

## Live demo

🔗 **[apex-ccws.streamlit.app](https://apex-ccws.streamlit.app)**

No login. No download. Opens in 15 seconds on any phone or computer.

## Tech stack

- **Frontend / app:** Streamlit (Python)
- **Smart search:** Llama 3.2 (local AI, privacy-first)
- **Data:** CSV-based service catalog with 40+ verified entries
- **Hosting:** Streamlit Community Cloud

## Project context

Built as the BSYS-4905 capstone consulting project for **Creating Community Wellness Society (CCWS)** in Victoria, BC.

## Team — APEX Consulting Group

| Member | Role |
|--------|------|
| Qian Li | Lead |
| Nilan Beigi | AI & Data |
| Sofiya Golagha | Research |
| Chaz Alec | Systems |

## Repository structure

- `app.py` — Main Streamlit application
- `services_dataset.csv` — Verified senior services data
- `generate_qr.py` — QR code generator for the live demo
- `requirements.txt` — Python dependencies
- `config.toml` — Streamlit configuration
- `.devcontainer/` — Development container setup
- `README_DEPLOY.md` — Deployment notes

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Course

BSYS-4905 · BCIT · Spring 2026
