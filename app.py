import re
import os
import json
import requests
import pandas as pd
import streamlit as st

# Optional voice search
try:
    from streamlit_mic_recorder import speech_to_text
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

st.set_page_config(
    page_title="CCWS Senior Services Demo",
    page_icon="💙",
    layout="wide",
)

@st.cache_data
def load_data():
    return pd.read_csv("services_dataset.csv")

df = load_data()

# ---------------- LLM Config ----------------
USE_OLLAMA = True
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ---------------- Session State ----------------
defaults = {
    "results": pd.DataFrame(),
    "recommended": pd.DataFrame(),
    "searched": False,
    "search_query": "",
    "voice_query": "",
    "final_query": "",
    "used_category": "All",
    "used_location": "All",
    "used_cost": "All",
    "used_engine_query": "",
    "assistant_reply": "",
    "assistant_summary": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Intent / Assistant Logic ----------------
CATEGORY_KEYWORDS = {
    "Housing": ["housing", "rent", "home", "shelter", "apartment", "living"],
    "Health": ["health", "doctor", "clinic", "care", "mental", "nursing", "medical"],
    "Community Support": ["support", "meals", "caregiver", "community", "help", "program", "senior"],
    "Transportation": ["transport", "bus", "ride", "transit", "handydart"],
    "Financial": ["money", "financial", "income", "benefit", "food bank", "subsidy", "subsidized"],
    "Safety": ["legal", "abuse", "safe", "emergency", "protection"],
}

LOCATION_KEYWORDS = ["victoria", "bc"]
COST_KEYWORDS = {
    "Free": ["free"],
    "Low-cost": ["low cost", "low-cost", "cheap", "affordable"],
    "Paid": ["paid", "private"],
    "Subsidized": ["subsidized", "subsidy"],
    "Public funded": ["public funded", "funded"],
}

def clean_user_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def extract_need_from_text(text: str):
    cleaned = clean_user_text(text)

    category_scores = {}
    for category, words in CATEGORY_KEYWORDS.items():
        score = sum(1 for w in words if w in cleaned)
        category_scores[category] = score

    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] == 0:
        best_category = None

    best_location = None
    for loc in LOCATION_KEYWORDS:
        if loc in cleaned:
            best_location = loc.title() if loc != "bc" else "BC"
            break

    best_cost = None
    for cost_label, words in COST_KEYWORDS.items():
        if any(w in cleaned for w in words):
            best_cost = cost_label
            break

    summary = None
    if best_category:
        if best_location and best_cost:
            summary = f"{best_category} support in {best_location} with {best_cost.lower()} options"
        elif best_location:
            summary = f"{best_category} support in {best_location}"
        elif best_cost:
            summary = f"{best_category} support with {best_cost.lower()} options"
        else:
            summary = f"{best_category} support"

    return {
        "category": best_category,
        "location": best_location,
        "cost": best_cost,
        "summary": summary
    }

def has_real_search(query_text, category, location, cost, engine_query):
    return bool(
        str(query_text).strip()
        or str(engine_query).strip()
        or category != "All"
        or location != "All"
        or cost != "All"
    )

def run_search(dataframe, query_text="", category="All", location="All", cost="All", engine_query=""):
    filtered = dataframe.copy()

    if category != "All":
        filtered = filtered[filtered["Category"] == category]
    if location != "All":
        filtered = filtered[filtered["Location"] == location]
    if cost != "All":
        filtered = filtered[filtered["Cost"] == cost]

    combined_query = " ".join([str(query_text).strip(), str(engine_query).strip()]).strip()

    if combined_query:
        mask = (
            filtered["Name"].str.contains(combined_query, case=False, na=False)
            | filtered["Description"].str.contains(combined_query, case=False, na=False)
            | filtered["Eligibility"].str.contains(combined_query, case=False, na=False)
            | filtered["Provider"].str.contains(combined_query, case=False, na=False)
            | filtered["Category"].str.contains(combined_query, case=False, na=False)
            | filtered["Location"].str.contains(combined_query, case=False, na=False)
            | filtered["Source"].str.contains(combined_query, case=False, na=False)
        )
        filtered = filtered[mask]

    return filtered

def score_results(results_df, query_text):
    if results_df.empty:
        return results_df

    scored = results_df.copy()
    scored["score"] = 0

    q = str(query_text).strip()
    if q:
        scored["score"] += scored["Name"].str.contains(q, case=False, na=False).astype(int) * 4
        scored["score"] += scored["Description"].str.contains(q, case=False, na=False).astype(int) * 3
        scored["score"] += scored["Category"].str.contains(q, case=False, na=False).astype(int) * 2
        scored["score"] += scored["Eligibility"].str.contains(q, case=False, na=False).astype(int) * 1

    return scored.sort_values(["score", "Name"], ascending=[False, True])

def build_service_context(results_df, max_rows=8):
    if results_df.empty:
        return "No matching services found."

    rows = []
    for _, row in results_df.head(max_rows).iterrows():
        rows.append({
            "Name": row["Name"],
            "Category": row["Category"],
            "Description": row["Description"],
            "Location": row["Location"],
            "Cost": row["Cost"],
            "Eligibility": row["Eligibility"],
            "Contact": row["Contact"],
            "Provider": row["Provider"],
            "Source": row["Source"],
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)

def call_ollama_chat(user_request, filtered_df):
    if not USE_OLLAMA:
        return None

    service_context = build_service_context(filtered_df)

    system_prompt = """
You are a helpful senior-services assistant for a proof-of-concept app.
Your job is to:
1. Read the user's request.
2. Look only at the provided service dataset candidates.
3. Recommend the 3 best services from the dataset.
4. Respond in simple, warm language for seniors.
5. Return STRICT JSON with this exact shape:
{
  "detected_need": "...",
  "assistant_reply": "...",
  "recommended_names": ["...", "...", "..."]
}
Rules:
- Do not invent services not in the dataset.
- If the user message is messy, extract the real need from it.
- Keep the reply short and useful.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User request:\n{user_request}\n\nCandidate services:\n{service_context}"
            }
        ],
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "{}")
        return json.loads(content)
    except Exception:
        return None

def get_top_recommendations_from_names(results_df, names):
    if not names:
        return results_df.head(3).to_dict("records")

    picked = []
    used = set()

    for wanted in names:
        matches = results_df[
            results_df["Name"].str.contains(str(wanted), case=False, na=False)
        ]
        for _, row in matches.iterrows():
            if row["Name"] not in used:
                picked.append(row.to_dict())
                used.add(row["Name"])
                break

    if len(picked) < 3:
        for _, row in results_df.iterrows():
            if row["Name"] not in used:
                picked.append(row.to_dict())
                used.add(row["Name"])
            if len(picked) == 3:
                break

    return picked[:3]

def assistant_search(dataframe, raw_text, manual_category, manual_location, manual_cost, engine_query):
    extracted = extract_need_from_text(raw_text)

    category = manual_category if manual_category != "All" else (extracted["category"] if extracted["category"] else "All")
    location = manual_location if manual_location != "All" else (extracted["location"] if extracted["location"] else "All")
    cost = manual_cost if manual_cost != "All" else (extracted["cost"] if extracted["cost"] else "All")

    query = raw_text.strip()
    results = run_search(
        dataframe,
        query_text=query,
        category=category,
        location=location,
        cost=cost,
        engine_query=engine_query
    )

    if results.empty:
        results = run_search(
            dataframe,
            query_text="",
            category=category,
            location=location,
            cost=cost,
            engine_query=engine_query
        )

    if results.empty:
        reply = (
            "I could not find a strong match in the current sample dataset. "
            "Try a simpler request like housing, meals, transportation, legal help, or senior support."
        )
        summary = extracted["summary"] if extracted["summary"] else "General support"
        return results, pd.DataFrame(), reply, summary

    llm_output = call_ollama_chat(raw_text, results)

    if llm_output:
        summary = llm_output.get("detected_need") or extracted["summary"] or "General support"
        reply = llm_output.get("assistant_reply") or "I found some matching services for your request."
        recommended_names = llm_output.get("recommended_names", [])
        top_recs = get_top_recommendations_from_names(results, recommended_names)
    else:
        combined_query = " ".join([query, engine_query]).strip()
        ranked = score_results(results, combined_query)
        top_recs = ranked.head(3).to_dict("records")
        top_names = [r["Name"] for r in top_recs]
        reply = (
            f"Based on your request, I identified the main need as: "
            f"{extracted['summary'] if extracted['summary'] else 'general support'}. "
            f"I found {len(results)} matching services. "
            f"My top recommendations are {', '.join(top_names)}."
        )
        summary = extracted["summary"] if extracted["summary"] else "General support"

    recommended_df = pd.DataFrame(top_recs)
    return results, recommended_df, reply, summary

# ---------------- Style ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.main {
    background: linear-gradient(180deg, #f5f9fc 0%, #edf4fa 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}
[data-testid="stSidebar"] {
    background: #dfeaf3;
    border-right: 1px solid #c7d7e5;
}
[data-testid="stSidebar"] * {
    color: #18354c !important;
}
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background-color: #ffffff !important;
    color: #18354c !important;
    border-radius: 12px !important;
}
.stButton > button {
    background-color: #4f8fb8 !important;
    color: white !important;
    border: 1px solid #3f7da5 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
}
.stButton > button:hover {
    background-color: #3f7da5 !important;
    color: white !important;
    border: 1px solid #356b8e !important;
}
.hero-wrap {
    background: linear-gradient(135deg, #dceaf7 0%, #f7fbff 100%);
    border: 1px solid #c6d8e8;
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #163a56;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1.08rem;
    color: #36556b;
    line-height: 1.7;
}
.search-panel {
    background: #f7fbff;
    border: 1px solid #cdddea;
    border-radius: 20px;
    padding: 1rem 1rem 0.6rem 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.search-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #163a56;
    margin-bottom: 0.5rem;
}
.small-note {
    color: #4c697f;
    font-size: 0.96rem;
}
.soft-card {
    background: #f7fbff;
    border: 1px solid #cdddea;
    border-radius: 18px;
    padding: 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    color: #163a56;
}
.metric-card {
    background: #f7fbff;
    border: 1px solid #d3e1ed;
    border-radius: 18px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
.metric-number {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0f4f7c;
}
.metric-label {
    color: #577487;
    font-size: 0.95rem;
}
.service-card {
    background: #f7fbff;
    border: 1px solid #cfe0ec;
    border-left: 6px solid #79afd1;
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    color: #173951;
}
.service-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #173951;
    margin-bottom: 0.25rem;
}
.service-meta {
    color: #577287;
    font-size: 0.94rem;
    margin-bottom: 0.7rem;
}
.tag {
    display: inline-block;
    background: #dcebf7;
    color: #1c577f;
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.82rem;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}
.quick-box {
    background: #eaf4fb;
    border: 1px solid #c9ddea;
    border-radius: 18px;
    padding: 1rem;
    color: #163a56;
    box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.assistant-box {
    background: #eef7f2;
    border: 1px solid #b8d9c1;
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    color: #1d4f36;
    box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.rec-card {
    background: #ffffff;
    border: 1px solid #b2d4e8;
    border-left: 5px solid #4f8fb8;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    color: #163a56;
}
.rec-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #163a56;
    margin-bottom: 0.2rem;
}
.rec-meta {
    font-size: 0.88rem;
    color: #4f7a94;
    margin-bottom: 0.4rem;
}
.rec-desc {
    font-size: 0.93rem;
    color: #2d4e62;
}
.section-note {
    color: #4c697f;
    font-size: 0.98rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.markdown("## 💙 Filter Services")
st.sidebar.markdown("Use filters or type below to search from the left side.")

categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
locations = ["All"] + sorted(df["Location"].dropna().unique().tolist())
costs = ["All"] + sorted(df["Cost"].dropna().unique().tolist())

selected_category = st.sidebar.selectbox(
    "Service Category",
    categories,
    index=categories.index(st.session_state.used_category) if st.session_state.used_category in categories else 0
)
selected_location = st.sidebar.selectbox(
    "Location",
    locations,
    index=locations.index(st.session_state.used_location) if st.session_state.used_location in locations else 0
)
selected_cost = st.sidebar.selectbox(
    "Cost",
    costs,
    index=costs.index(st.session_state.used_cost) if st.session_state.used_cost in costs else 0
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔍 Search engine")
engine_query = st.sidebar.text_input(
    "Search services",
    value=st.session_state.used_engine_query,
    placeholder="e.g. meals, housing, transport",
    label_visibility="collapsed"
)
sidebar_search_clicked = st.sidebar.button("Apply Filters & Search", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌟 Future Accessibility Ideas")
st.sidebar.markdown(
    "- Voice-assisted search\n"
    "- Larger text mode\n"
    "- Simpler caregiver mode"
)

# ---------------- Header ----------------
left, right = st.columns([1.4, 1])
with left:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">CCWS Senior Services Demo</div>
        <div class="hero-sub">
            A warmer, easier way to search for senior services.<br><br>
            This proof-of-concept shows how community support information could be brought together
            into one place for seniors, caregivers, and service providers.
        </div>
    </div>
    """, unsafe_allow_html=True)
with right:
    st.image(
        "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?auto=format&fit=crop&w=1200&q=80",
        use_container_width=True,
        caption="Friendly and accessible support experience"
    )

# ---------------- Main Search Area ----------------
st.markdown("""
<div class="search-panel">
    <div class="search-title">Search for support</div>
    <div class="small-note">Type what you need in plain language — the assistant will understand and recommend the best options.</div>
</div>
""", unsafe_allow_html=True)

search_col, voice_col = st.columns([3, 1])

with search_col:
    typed_query = st.text_input(
        "Search",
        value=st.session_state.search_query,
        placeholder='Try: "I need affordable housing in Victoria" or "free meals for seniors"',
        label_visibility="collapsed"
    )

with voice_col:
    voice_text = ""
    st.markdown("##### 🎤 Voice")
    if VOICE_AVAILABLE:
        voice_text = speech_to_text(
            language="en",
            start_prompt="Start voice",
            stop_prompt="Stop",
            just_once=True,
            use_container_width=True,
            key="main_voice_search",
        )
        if voice_text:
            st.success(f"Captured: {voice_text}")
    else:
        st.info("Voice package not installed.")

btn1, btn2 = st.columns([1, 1])
with btn1:
    search_clicked = st.button("Search Services", use_container_width=True)
with btn2:
    reset_clicked = st.button("Reset Search", use_container_width=True)

# ---------------- Execute Search ----------------
if search_clicked:
    final_query = voice_text.strip() if voice_text else typed_query.strip()

    st.session_state.search_query = typed_query
    st.session_state.voice_query = voice_text if voice_text else ""
    st.session_state.final_query = final_query
    st.session_state.used_category = selected_category
    st.session_state.used_location = selected_location
    st.session_state.used_cost = selected_cost
    st.session_state.used_engine_query = engine_query

    if has_real_search(final_query, selected_category, selected_location, selected_cost, engine_query):
        results, recommended, reply, summary = assistant_search(
            df,
            final_query,
            selected_category,
            selected_location,
            selected_cost,
            engine_query
        )
        st.session_state.results = results
        st.session_state.recommended = recommended
        st.session_state.assistant_reply = reply
        st.session_state.assistant_summary = summary
        st.session_state.searched = True
    else:
        st.session_state.results = pd.DataFrame()
        st.session_state.recommended = pd.DataFrame()
        st.session_state.assistant_reply = ""
        st.session_state.assistant_summary = ""
        st.session_state.searched = False

if sidebar_search_clicked:
    final_query = engine_query.strip()

    st.session_state.search_query = ""
    st.session_state.voice_query = ""
    st.session_state.final_query = final_query
    st.session_state.used_category = selected_category
    st.session_state.used_location = selected_location
    st.session_state.used_cost = selected_cost
    st.session_state.used_engine_query = engine_query

    if has_real_search(final_query, selected_category, selected_location, selected_cost, engine_query):
        results, recommended, reply, summary = assistant_search(
            df,
            final_query,
            selected_category,
            selected_location,
            selected_cost,
            engine_query
        )
        st.session_state.results = results
        st.session_state.recommended = recommended
        st.session_state.assistant_reply = reply
        st.session_state.assistant_summary = summary
        st.session_state.searched = True
    else:
        st.session_state.results = pd.DataFrame()
        st.session_state.recommended = pd.DataFrame()
        st.session_state.assistant_reply = ""
        st.session_state.assistant_summary = ""
        st.session_state.searched = False

if reset_clicked:
    st.session_state.results = pd.DataFrame()
    st.session_state.recommended = pd.DataFrame()
    st.session_state.searched = False
    st.session_state.final_query = ""
    st.session_state.search_query = ""
    st.session_state.voice_query = ""
    st.session_state.assistant_reply = ""
    st.session_state.assistant_summary = ""
    st.session_state.used_category = "All"
    st.session_state.used_location = "All"
    st.session_state.used_cost = "All"
    st.session_state.used_engine_query = ""

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Overview", "Search Results", "Platform & Cost"])

with tab1:
    st.markdown("## Overview")
    st.markdown("<div class='section-note'>This demo shows the direction of a future searchable senior services platform.</div><br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-number'>{len(df)}</div><div class='metric-label'>Sample services</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-number'>{df['Category'].nunique()}</div><div class='metric-label'>Categories</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-number'>{df['Location'].nunique()}</div><div class='metric-label'>Locations</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-number'>{df['Source'].nunique()}</div><div class='metric-label'>Sources</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("""
        <div class="soft-card">
            <h3>What this demo proves</h3>
            <p>This prototype shows how senior services could be organized in one place with clearer structure, simpler search, and a more welcoming experience.</p>
            <p><strong>What we completed this week</strong></p>
            <ul>
                <li>Researched BC211, government, and community service sources</li>
                <li>Created a structured sample dataset</li>
                <li>Defined the key fields for a future searchable system</li>
                <li>Built a lightweight proof-of-concept in Streamlit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="soft-card">
            <h3>Dataset structure</h3>
            <p>
            • Service name<br>
            • Category<br>
            • Description<br>
            • Location<br>
            • Cost<br>
            • Eligibility<br>
            • Contact<br>
            • Provider<br>
            • Source
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("## Search Results")

    if st.session_state.searched:
        if st.session_state.final_query:
            st.markdown(
                f"<div class='section-note'><strong>Your request:</strong> {st.session_state.final_query}</div>",
                unsafe_allow_html=True
            )
        if st.session_state.assistant_summary:
            st.markdown(
                f"<div class='section-note'><strong>Detected need:</strong> {st.session_state.assistant_summary}</div><br>",
                unsafe_allow_html=True
            )
    else:
        st.info("Use the search area above or the search engine on the left, then click Search Services.")

    if st.session_state.assistant_reply:
        st.markdown(f"""
        <div class="assistant-box">
            <strong>🤖 Assistant Response</strong><br><br>
            {st.session_state.assistant_reply}
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.recommended.empty:
        st.markdown("### ⭐ Recommended options")
        st.markdown("<div class='section-note'>Based on your request, these are the best matching services:</div><br>", unsafe_allow_html=True)
        rec_cols = st.columns(min(3, len(st.session_state.recommended)))
        for i, (_, row) in enumerate(st.session_state.recommended.iterrows()):
            with rec_cols[i]:
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">⭐ {row['Name']}</div>
                    <div class="rec-meta">{row['Category']} • {row['Location']} • {row['Cost']}</div>
                    <div class="rec-desc">{row['Description']}</div>
                    <div style="margin-top:0.5rem;">
                        <span class="tag">Eligibility: {row['Eligibility']}</span>
                        <span class="tag">Contact: {row['Contact']}</span>
                        <span class="tag">Provider: {row['Provider']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")

    results = st.session_state.results

    if not results.empty:
        st.markdown(f"<div class='section-note'>All results found: <strong>{len(results)}</strong></div><br>", unsafe_allow_html=True)

        for _, row in results.iterrows():
            st.markdown(f"""
            <div class="service-card">
                <div class="service-title">{row['Name']}</div>
                <div class="service-meta">{row['Category']} • {row['Location']} • {row['Cost']}</div>
                <div style="margin-bottom:0.8rem;">{row['Description']}</div>
                <span class="tag">Eligibility: {row['Eligibility']}</span>
                <span class="tag">Provider: {row['Provider']}</span>
                <span class="tag">Source: {row['Source']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Service details")
        selected_service = st.selectbox("Choose a service for full details", results["Name"].tolist())
        row = results[results["Name"] == selected_service].iloc[0]

        left, right = st.columns([1.5, 1])
        with left:
            st.markdown(f"## {row['Name']}")
            st.write(row["Description"])
            st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"**Location:** {row['Location']}")
            st.markdown(f"**Eligibility:** {row['Eligibility']}")
        with right:
            st.markdown("""
            <div class="quick-box">
                <h3>Quick facts</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Cost:** {row['Cost']}")
            st.markdown(f"**Contact:** {row['Contact']}")
            st.markdown(f"**Provider:** {row['Provider']}")
            st.markdown(f"**Source:** {row['Source']}")
    elif st.session_state.searched:
        st.warning("No services match your search. Try different keywords or reset filters.")

with tab3:
    st.markdown("## Platform & Cost")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("""
        <div class="soft-card">
            <h3>Recommended platform direction</h3>
            <p>For the proof-of-concept, the best choice is a lightweight standalone demo platform, not direct modification of the CCWS website yet.</p>
            <p><strong>Recommended stack</strong></p>
            <ul>
                <li>Platform: Streamlit</li>
                <li>Data source: Excel / CSV structured dataset</li>
                <li>Future path: verify and expand data, then decide on full implementation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="soft-card">
            <h3>How the estimate was built</h3>
            <p>This rough estimate is based on a small nonprofit service platform with a simple search interface, structured database, design setup, hosting, and ongoing human data verification.</p>
            <ul>
                <li>Small proof-of-concept build first</li>
                <li>Later production version with better design and deployment</li>
                <li>Ongoing maintenance depends more on data updates than code</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="soft-card">
            <h3>Rough future cost estimate</h3>
        </div>
        """, unsafe_allow_html=True)

        cost_df = pd.DataFrame(
            {
                "Item": [
                    "UI design + front-end setup",
                    "Search/database setup",
                    "Testing + deployment setup",
                    "Hosting and maintenance (annual)",
                    "Human data verification / updates (annual)",
                ],
                "Estimate": [
                    "$4,000 - $6,000",
                    "$4,000 - $6,000",
                    "$2,000 - $4,000",
                    "$500 - $1,500",
                    "$5,000 - $10,000",
                ],
            }
        )
        st.table(cost_df)

    st.markdown("### Next steps")
    st.markdown("""
    1. Refine and verify the sample dataset  
    2. Improve source tracking  
    3. Expand service coverage  
    4. Improve the interface further  
    5. Add accessibility features like voice search after the core search flow is stable  
    """)