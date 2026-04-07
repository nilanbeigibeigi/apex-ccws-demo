import re
import os
import json
import uuid
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

# ---------------- Optional voice search ----------------
try:
    from streamlit_mic_recorder import speech_to_text
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(
    page_title="CCWS Senior Services Directory",
    page_icon="💙",
    layout="wide",
)

# ---------------- Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("services_dataset.csv")

    rename_map = {
        "Name": "ServiceName",
        "Category": "Category",
        "Description": "Description",
        "Location": "Location",
        "Cost": "Cost",
        "Eligibility": "Eligibility",
        "Contact": "Contact",
        "Provider": "Provider",
        "Source": "Source",
    }
    df = df.rename(columns=rename_map)

    required_cols = [
        "ServiceName",
        "Category",
        "Description",
        "Location",
        "Eligibility",
        "Source",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    if "ServiceID" not in df.columns:
        df["ServiceID"] = [f"SVC-{i+1:04d}" for i in range(len(df))]

    if "FriendlyCategory" not in df.columns:
        df["FriendlyCategory"] = df["Category"]

    if "Subcategory" not in df.columns:
        df["Subcategory"] = ""

    # New location fields from Joyce
    if "AreaOfVictoria" not in df.columns:
        df["AreaOfVictoria"] = "Victoria"
    if "TransitAccess" not in df.columns:
        df["TransitAccess"] = "Unknown"
    if "AccessMode" not in df.columns:
        df["AccessMode"] = "In person"

    # New contact fields from Joyce
    if "ContactPerson" not in df.columns:
        df["ContactPerson"] = ""
    if "Organization" not in df.columns:
        df["Organization"] = df["Provider"] if "Provider" in df.columns else ""
    if "Website" not in df.columns:
        df["Website"] = ""
    if "Phone" not in df.columns:
        df["Phone"] = ""
    if "Email" not in df.columns:
        df["Email"] = ""
    if "ChatAvailable" not in df.columns:
        df["ChatAvailable"] = "No"
    if "FAQAvailable" not in df.columns:
        df["FAQAvailable"] = "No"

    # New fee wording from Joyce
    if "FeeType" not in df.columns:
        df["FeeType"] = "Free"

    # Verification / maintenance
    if "VerificationStatus" not in df.columns:
        df["VerificationStatus"] = "AI reviewed"
    if "LastVerified" not in df.columns:
        df["LastVerified"] = "2026-04-01"
    if "LastUpdatedByProvider" not in df.columns:
        df["LastUpdatedByProvider"] = ""
    if "IsSubscriber" not in df.columns:
        df["IsSubscriber"] = "No"

    # New user experience fields
    if "UserRating" not in df.columns:
        df["UserRating"] = 0
    if "FriendlyScore" not in df.columns:
        df["FriendlyScore"] = 0
    if "EfficientScore" not in df.columns:
        df["EfficientScore"] = 0
    if "EasyToUnderstandScore" not in df.columns:
        df["EasyToUnderstandScore"] = 0
    if "AccessibilityScore" not in df.columns:
        df["AccessibilityScore"] = 0

    return df

base_df = load_data()

# ---------------- LLM Config ----------------
USE_OLLAMA = True
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ---------------- Session State ----------------
defaults = {
    "services_df": base_df.copy(),
    "results": pd.DataFrame(),
    "recommended": pd.DataFrame(),
    "searched": False,
    "search_query": "",
    "voice_query": "",
    "final_query": "",
    "voice_text_display": "",
    "assistant_reply": "",
    "assistant_summary": "",
    "used_category": "All",
    "used_location": "All",
    "used_cost": "All",
    "used_access_mode": "All",
    "used_engine_query": "",
    "experience_mode": "Guest / Public",
    "provider_logged_in": False,
    "provider_name_demo": "",
    "user_logged_in": False,
    "user_name_demo": "",
    "user_role_demo": "Senior",
    "saved_services": [],
    "active_tab": 0,
    "user_requests": pd.DataFrame(columns=[
        "RequestID",
        "UserName",
        "UserRole",
        "ServiceID",
        "ServiceName",
        "Provider",
        "Message",
        "PreferredContact",
        "Status",
        "CreatedAt",
    ]),
    "user_profile": {
        "full_name": "",
        "role": "Senior",
        "location": "Victoria",
        "preferred_contact": "Phone",
        "cost_preference": "Public",
        "mobility_needs": "",
        "transportation_needs": "",
        "support_interests": [],
        "notes": "",
    },
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Language / Legend Logic ----------------
TERM_EQUIVALENTS = {
    "primary care provider": ["doctor", "family doctor", "medical clinic", "nurse practitioner", "clinic"],
    "longevity": ["aging", "healthy aging", "senior wellness", "community support"],
    "community support": ["senior help", "daily help", "meals", "friendly visits", "caregiver support"],
    "food and nutrition": ["food", "meals", "nutrition", "prepared meals", "meal delivery", "grocery"],
    "housing": ["seniors housing", "rent help", "shelter", "home support", "rental"],
    "transportation": ["rides", "bus", "transit", "handydart", "taxi", "driver", "ride", "transport"],
    "income support": ["benefits", "money help", "financial aid", "subsidy", "income assistance"],
    "safety": ["legal help", "abuse support", "protection", "emergency help"],
}

USER_FRIENDLY_LEGEND = [
    ("Housing", "Housing support, rent help, shelter, accessible living"),
    ("Income Support", "Benefits, money help, subsidy, income assistance"),
    ("Health & Home Care", "Doctors, clinics, home care, health support"),
    ("Emergency", "Urgent support, crisis help, emergency contacts"),
    ("Food & Nutrition", "Meal delivery, food access, nutrition support"),
    ("Social Activities", "Community groups, social events, friendly activities"),
    ("Taxes & Administrative Support", "Tax help, forms, admin help, applications"),
    ("Mobility", "Walking aids, mobility support, accessibility assistance"),
    ("Transportation", "Bus, rides, transit, HandyDART"),
    ("Recreation", "Exercise, hobbies, wellness, recreation programs"),
    ("Education", "Learning programs, workshops, information sessions"),
]

PROFILE_SHORTCUTS = {
    "I need a doctor": "doctor clinic medical",
    "I need food help": "food meals nutrition prepared meals",
    "I need housing help": "housing rent shelter",
    "I need transportation": "transportation rides bus handydart",
    "I need caregiver support": "caregiver community support friendly visits",
    "I need money help": "benefits income support subsidy financial help",
}

CATEGORY_KEYWORDS = {
    "Housing": ["housing", "rent", "shelter", "home support", "accessible living", "rental", "affordable housing"],
    "Income Support": ["income", "benefits", "financial", "subsidy", "money help", "financial aid", "energy", "utility"],
    "Health & Home Care": ["doctor", "clinic", "health", "home care", "nurse", "physio", "care", "medical", "dental", "mental health", "respite", "caregiver"],
    "Emergency": ["emergency", "urgent", "crisis", "immediate help", "safety", "abuse", "legal", "alert"],
    "Food & Nutrition": ["food", "nutrition", "meals", "meal delivery", "prepared meals", "grocery", "food bank", "hamper"],
    "Social Activities": ["social", "activities", "community", "friendly visits", "social support", "dementia", "peer support", "visitor", "companionship"],
    "Taxes & Administrative Support": ["tax", "administrative", "paperwork", "forms", "applications", "filing"],
    "Mobility": ["mobility", "walker", "wheelchair", "accessibility", "movement support", "cane", "equipment"],
    "Transportation": ["transportation", "bus", "ride", "rides", "transit", "handydart", "taxi", "driver", "transport", "accessible transit"],
    "Recreation": ["recreation", "exercise", "wellness", "hobbies", "fitness", "yoga", "aquafit", "sport"],
    "Education": ["education", "learning", "class", "workshop", "training", "computer", "literacy"],
}

COST_KEYWORDS = {
    "Free": ["free", "no cost", "fully funded"],
    "Partial pay": ["partial pay", "partial", "subsidized", "partially funded"],
    "Subscription": ["subscription", "monthly fee", "member fee"],
    "Private pay": ["private pay", "paid", "fee for service", "private"],
}

STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "want", "looking", "look", "the", "a", "an", "and", "or",
    "to", "in", "on", "at", "with", "find",
    "show", "please", "some", "is", "are", "am", "of", "near", "around",
    "get", "can", "do", "would", "like", "have", "has"
}

def clean_user_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_query(text: str):
    cleaned = clean_user_text(text)
    return [t for t in cleaned.split() if t not in STOP_WORDS and len(t) > 1]

def expand_query_terms(text: str):
    cleaned = clean_user_text(text)
    expanded = set(tokenize_query(cleaned))

    # Also add the raw words from the query (before stop word removal) so short keywords still match
    for word in cleaned.split():
        if len(word) > 2:
            expanded.add(word)

    for key, synonyms in TERM_EQUIVALENTS.items():
        if key in cleaned:
            expanded.add(clean_user_text(key))
            expanded.update(tokenize_query(" ".join(synonyms)))
        for synonym in synonyms:
            if clean_user_text(synonym) in cleaned:
                expanded.add(clean_user_text(key))
                expanded.update(tokenize_query(" ".join(synonyms)))

    # If query matches a category keyword group, add all synonyms from that group
    for category, words in CATEGORY_KEYWORDS.items():
        if any(w in cleaned for w in words):
            expanded.update(words)

    return [x for x in expanded if x and len(x) > 1]

def extract_need_from_text(text: str):
    cleaned = clean_user_text(text)

    category_scores = {}
    for category, words in CATEGORY_KEYWORDS.items():
        category_scores[category] = sum(1 for w in words if w in cleaned)

    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] == 0:
        best_category = None

    best_cost = None
    for cost_label, words in COST_KEYWORDS.items():
        if any(w in cleaned for w in words):
            best_cost = cost_label
            break

    best_location = "Victoria" if "victoria" in cleaned else None

    summary = None
    if best_category:
        if best_location and best_cost:
            summary = f"{best_category} in {best_location} with {best_cost.lower()} options"
        elif best_location:
            summary = f"{best_category} in {best_location}"
        elif best_cost:
            summary = f"{best_category} with {best_cost.lower()} options"
        else:
            summary = best_category

    return {
        "category": best_category,
        "location": best_location,
        "cost": best_cost,
        "summary": summary,
    }

def normalize_friendly_category(raw_category: str, description: str = "") -> str:
    # If the category already matches a known label exactly, return it as-is
    known_labels = list(CATEGORY_KEYWORDS.keys())
    if str(raw_category).strip() in known_labels:
        return str(raw_category).strip()
    combined = clean_user_text(f"{raw_category} {description}")
    for label, keywords in CATEGORY_KEYWORDS.items():
        if any(k in combined for k in keywords):
            return label
    return raw_category if str(raw_category).strip() else "Community Support"

def normalize_cost_type(value: str):
    v = clean_user_text(value)
    if any(x in v for x in ["free", "no cost", "fully funded"]):
        return "Free"
    if any(x in v for x in ["partial pay", "partial", "subsidized", "partially funded"]):
        return "Partial pay"
    if any(x in v for x in ["subscription", "monthly fee", "member fee"]):
        return "Subscription"
    if any(x in v for x in ["private", "paid", "fee", "fee for service", "private pay"]):
        return "Private pay"
    return "Private pay"

st.session_state.services_df["FriendlyCategory"] = st.session_state.services_df.apply(
    lambda row: normalize_friendly_category(row.get("Category", ""), row.get("Description", "")),
    axis=1
)
st.session_state.services_df["FeeType"] = st.session_state.services_df["FeeType"].apply(normalize_cost_type)
# ---------------- Search ----------------
def has_real_search(query_text, category, location, cost, access_mode, engine_query):
    return bool(
        str(query_text).strip()
        or str(engine_query).strip()
        or category != "All"
        or location != "All"
        or cost != "All"
        or access_mode != "All"
    )

def build_search_blob(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["ServiceName"].fillna("") + " " +
        frame["FriendlyCategory"].fillna("") + " " +
        frame["Subcategory"].fillna("") + " " +
        frame["Description"].fillna("") + " " +
        frame["Eligibility"].fillna("") + " " +
        frame["Provider"].fillna("") + " " +
        frame["Organization"].fillna("") + " " +
        frame["ContactPerson"].fillna("") + " " +
        frame["Location"].fillna("") + " " +
        frame["AreaOfVictoria"].fillna("") + " " +
        frame["TransitAccess"].fillna("") + " " +
        frame["AccessMode"].fillna("") + " " +
        frame["Phone"].fillna("") + " " +
        frame["Email"].fillna("") + " " +
        frame["Source"].fillna("")
    ).str.lower()

def run_search(dataframe, query_text="", category="All", location="All", cost=None, access_mode="All", engine_query=""):
    filtered = dataframe.copy()

    if category != "All":
        filtered = filtered[filtered["FriendlyCategory"] == category]

    if location != "All":
        filtered = filtered[filtered["Location"] == location]

    if cost and cost != "All":
        filtered = filtered[filtered["FeeType"] == cost]

    if access_mode != "All":
        filtered = filtered[filtered["AccessMode"] == access_mode]

    combined_query = " ".join([str(query_text).strip(), str(engine_query).strip()]).strip()
    terms = expand_query_terms(combined_query)

    if terms:
        blob = build_search_blob(filtered)
        mask = pd.Series(False, index=filtered.index)
        for term in terms:
            mask = mask | blob.str.contains(re.escape(term), case=False, na=False)
        keyword_results = filtered[mask]

        # If keyword search found results, return them
        if not keyword_results.empty:
            return keyword_results

        # Fallback 1: try category-only if a category was detected
        extracted = extract_need_from_text(combined_query)
        detected_cat = extracted.get("category")
        if detected_cat and detected_cat != "All":
            cat_results = dataframe.copy()
            if location != "All":
                cat_results = cat_results[cat_results["Location"] == location]
            cat_results = cat_results[cat_results["FriendlyCategory"] == detected_cat]
            if not cat_results.empty:
                return cat_results

        # Fallback 2: return all filtered (by sidebar filters only) if no keyword match
        return filtered

    return filtered

def score_results(results_df, query_text):
    if results_df.empty:
        return results_df

    scored = results_df.copy()
    scored["score"] = 0
    terms = expand_query_terms(query_text)

    for t in terms:
        t_re = re.escape(t)
        scored["score"] += scored["ServiceName"].fillna("").str.lower().str.contains(t_re, regex=True).astype(int) * 5
        scored["score"] += scored["Subcategory"].fillna("").str.lower().str.contains(t_re, regex=True).astype(int) * 4
        scored["score"] += scored["FriendlyCategory"].fillna("").str.lower().str.contains(t_re, regex=True).astype(int) * 3
        scored["score"] += scored["Description"].fillna("").str.lower().str.contains(t_re, regex=True).astype(int) * 2
        scored["score"] += scored["Eligibility"].fillna("").str.lower().str.contains(t_re, regex=True).astype(int) * 1

    verification_bonus = {
        "Provider updated": 3,
        "Verified": 2,
        "AI reviewed": 1,
        "Needs review": 0,
    }
    scored["verify_bonus"] = scored["VerificationStatus"].map(verification_bonus).fillna(0)
    scored["final_score"] = scored["score"] + scored["verify_bonus"]

    return scored.sort_values(["final_score", "ServiceName"], ascending=[False, True])

def build_service_context(results_df, max_rows=8):
    if results_df.empty:
        return "No matching services found."

    rows = []
    for _, row in results_df.head(max_rows).iterrows():
        rows.append({
            "ServiceName": row["ServiceName"],
            "FriendlyCategory": row["FriendlyCategory"],
            "Subcategory": row["Subcategory"],
            "Description": row["Description"],
            "Location": row["Location"],
            "CostType": row.get("FeeType", row.get("CostType", "")),
            "Eligibility": row["Eligibility"],
            "Contact": row["Contact"],
            "Provider": row["Provider"],
            "Source": row["Source"],
            "VerificationStatus": row["VerificationStatus"],
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
2. Translate confusing healthcare/government terms into simple language.
3. Look only at the provided service dataset candidates.
4. Recommend the 3 best services from the dataset.
5. Respond in simple, warm language for seniors and families.
6. Return STRICT JSON with this exact shape:
{
  "detected_need": "...",
  "assistant_reply": "...",
  "recommended_names": ["...", "...", "..."]
}
Rules:
- Do not invent services not in the dataset.
- Focus on service-first matching, not provider-first.
- If the user says "longevity", connect it to aging / senior wellness / community support.
- Keep the reply short and useful.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User request:\\n{user_request}\\n\\nCandidate services:\\n{service_context}"
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
        return results_df.head(3).copy()

    picked = []
    used = set()

    for wanted in names:
        matches = results_df[
            results_df["ServiceName"].str.contains(str(wanted), case=False, na=False, regex=False)
        ]
        for _, row in matches.iterrows():
            if row["ServiceID"] not in used:
                picked.append(row.to_dict())
                used.add(row["ServiceID"])
                break

    if len(picked) < 3:
        for _, row in results_df.iterrows():
            if row["ServiceID"] not in used:
                picked.append(row.to_dict())
                used.add(row["ServiceID"])
            if len(picked) == 3:
                break

    return pd.DataFrame(picked[:3])

def assistant_search(dataframe, raw_text, manual_category, manual_location, manual_cost, manual_access_mode, engine_query):
    extracted = extract_need_from_text(raw_text)

    category = manual_category if manual_category != "All" else (extracted["category"] if extracted["category"] else "All")
    location = manual_location if manual_location != "All" else (extracted["location"] if extracted["location"] else "All")
    cost = manual_cost if manual_cost != "All" else (extracted["cost"] if extracted["cost"] else "All")
    access_mode = manual_access_mode

    query = raw_text.strip()
    results = run_search(
        dataframe,
        query_text=query,
        category=category,
        location=location,
        cost=cost,
        access_mode=access_mode,
        engine_query=engine_query
    )

    # If still empty, widen search - drop location and cost filters
    if results.empty:
        results = run_search(
            dataframe,
            query_text=query,
            category=category,
            location="All",
            cost="All",
            access_mode="All",
            engine_query=engine_query
        )

    # If still empty, return everything in the detected category
    if results.empty and extracted["category"]:
        results = dataframe[dataframe["FriendlyCategory"] == extracted["category"]].copy()

    # Last resort - return all services
    if results.empty:
        results = dataframe.copy()
        reply = (
            "I couldn't find an exact match, but here are all available services. "
            "Try words like: doctor, meals, housing, transportation, bus, handydart, food, caregiver, legal, tax, or fitness."
        )
        summary = "All services"
        return results, results.head(3).copy(), reply, summary

    ranked = score_results(results, " ".join([query, engine_query]).strip())
    llm_output = call_ollama_chat(raw_text, ranked)

    if llm_output:
        summary = llm_output.get("detected_need") or extracted["summary"] or "General support"
        reply = llm_output.get("assistant_reply") or "I found some matching services for your request."
        recommended_names = llm_output.get("recommended_names", [])
        top_recs = get_top_recommendations_from_names(ranked, recommended_names)
    else:
        top_recs = ranked.head(3).copy()
        top_names = top_recs["ServiceName"].tolist()
        reply = (
            f"Based on your request, I found services related to "
            f"{extracted['summary'] if extracted['summary'] else 'your need'}. "
            f"My top recommendations are: {', '.join(top_names)}."
        )
        summary = extracted["summary"] if extracted["summary"] else "General support"

    return ranked, top_recs, reply, summary

# ---------------- Save / Request helpers ----------------
def save_service(service_id: str):
    if service_id not in st.session_state.saved_services:
        st.session_state.saved_services.append(service_id)

def remove_saved_service(service_id: str):
    st.session_state.saved_services = [x for x in st.session_state.saved_services if x != service_id]

def add_user_request(service_row, message, preferred_contact):
    request_row = {
        "RequestID": f"REQ-{uuid.uuid4().hex[:8].upper()}",
        "UserName": st.session_state.user_profile["full_name"] or st.session_state.user_name_demo or "Guest User",
        "UserRole": st.session_state.user_profile["role"],
        "ServiceID": service_row["ServiceID"],
        "ServiceName": service_row["ServiceName"],
        "Provider": service_row["Provider"],
        "Message": message,
        "PreferredContact": preferred_contact,
        "Status": "Pending",
        "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.user_requests = pd.concat(
        [st.session_state.user_requests, pd.DataFrame([request_row])],
        ignore_index=True
    )

# ---------------- Provider add/update ----------------
def add_provider_service(
    service_name,
    category,
    subcategory,
    organization,
    contact_person,
    location,
    area_of_victoria,
    transit_access,
    access_mode,
    website,
    phone,
    email,
    communication_options,
    fee_type,
    description,
    rating,
    friendly,
    efficient,
    easy_to_understand,
    accessibility,
):
    import pandas as pd
    from datetime import datetime

    chat_available = "Yes" if "Chat" in communication_options else "No"
    faq_available = "Yes" if "FAQ" in communication_options else "No"
    normalized_fee = normalize_cost_type(fee_type)

    new_row = {
        "ServiceID": f"SVC-P-{uuid.uuid4().hex[:6].upper()}",
        "ServiceName": service_name,
        "Category": category,
        "FriendlyCategory": category,
        "Subcategory": subcategory,
        "Provider": organization,
        "Organization": organization,
        "ContactPerson": contact_person,
        "Location": location,
        "AreaOfVictoria": area_of_victoria,
        "TransitAccess": transit_access,
        "AccessMode": access_mode,
        "Website": website,
        "Phone": phone,
        "Email": email,
        "ChatAvailable": chat_available,
        "FAQAvailable": faq_available,
        "FeeType": normalized_fee,
        "Description": description,
        "Eligibility": "",
        "Source": "Provider submitted",
        "UserRating": rating,
        "FriendlyScore": friendly,
        "EfficientScore": efficient,
        "EasyToUnderstandScore": easy_to_understand,
        "AccessibilityScore": accessibility,
        "VerificationStatus": "Provider updated",
        "LastVerified": datetime.now().strftime("%Y-%m-%d"),
        "LastUpdatedByProvider": datetime.now().strftime("%Y-%m-%d"),
        "IsSubscriber": "Yes",
        "Contact": phone or email,
    }

    if "services_df" not in st.session_state:
        st.session_state.services_df = pd.DataFrame()

    st.session_state.services_df = pd.concat(
        [st.session_state.services_df, pd.DataFrame([new_row])],
        ignore_index=True
    )

def verification_label(row):
    status = str(row.get("VerificationStatus", "")).strip()
    last_verified = str(row.get("LastVerified", "")).strip()
    if status == "Provider updated":
        return f"✅ Provider updated • last checked {last_verified}"
    if status == "Verified":
        return f"✅ Verified • last checked {last_verified}"
    if status == "AI reviewed":
        return f"🤖 AI reviewed • last checked {last_verified}"
    return f"⚠ Needs review • last checked {last_verified}"

def stars_html(score, max_score=5):
    """Return filled/empty star string for a 0-5 score."""
    try:
        val = float(score)
    except (ValueError, TypeError):
        val = 0
    val = max(0, min(val, max_score))
    filled = int(round(val))
    empty = max_score - filled
    return "★" * filled + "☆" * empty

def rating_block_html(row):
    """Build the full star-rating HTML block for a service row."""
    overall = float(row.get("UserRating", 0) or 0)
    friendly = float(row.get("FriendlyScore", 0) or 0)
    efficient = float(row.get("EfficientScore", 0) or 0)
    easy = float(row.get("EasyToUnderstandScore", 0) or 0)
    access = float(row.get("AccessibilityScore", 0) or 0)

    if overall == 0 and friendly == 0 and efficient == 0 and easy == 0 and access == 0:
        return '<div class="star-row"><span class="star-label">No ratings yet</span></div>'

    def bar(val):
        pct = int((val / 5) * 100)
        return (
            '<div class="rbar-wrap">'
            '<div class="rbar" style="width:' + str(pct) + '%"></div>'
            '</div>'
        )

    html = (
        '<div class="star-row">'
        '<span class="stars">' + stars_html(overall) + '</span>'
        '<span class="star-num">' + str(round(overall, 1)) + '/5</span>'
        '<span class="star-label">Overall rating</span>'
        '</div>'
        '<div class="rating-grid">'
        '<div class="rating-item"><span style="min-width:72px">&#128578; Friendly</span>' + bar(friendly) + '<span class="star-num">' + str(int(friendly)) + '/5</span></div>'
        '<div class="rating-item"><span style="min-width:72px">&#9889; Efficient</span>' + bar(efficient) + '<span class="star-num">' + str(int(efficient)) + '/5</span></div>'
        '<div class="rating-item"><span style="min-width:72px">&#128218; Clear</span>' + bar(easy) + '<span class="star-num">' + str(int(easy)) + '/5</span></div>'
        '<div class="rating-item"><span style="min-width:72px">&#9855; Accessible</span>' + bar(access) + '<span class="star-num">' + str(int(access)) + '/5</span></div>'
        '</div>'
    )
    return html

# ---------------- Style ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', 'Segoe UI', sans-serif;
    font-size: 16px;
}
.main {
    background: linear-gradient(160deg, #FFF8F0 0%, #FFF3E8 40%, #F0F7FF 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1380px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A4B6E 0%, #0F3450 100%);
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #E8F4FF !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] h2 {
    color: #FFD580 !important;
    font-size: 1.2rem !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1a3a50 !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    border: 2px solid #B8D4E8 !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: #4f8fb8 !important;
    box-shadow: 0 0 0 3px rgba(79,143,184,0.18) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #2E86C1, #1A5276) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(46,134,193,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1A6FA8, #0F3D5C) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(46,134,193,0.4) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.7);
    border-radius: 16px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #D6E8F5;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.97rem !important;
    color: #2d5a7a !important;
    padding: 0.5rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2E86C1, #1A5276) !important;
    color: white !important;
}

/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #1A4B6E 0%, #2E86C1 100%);
    border-radius: 24px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 12px 32px rgba(26,75,110,0.25);
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: "💙";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.12;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.85);
    line-height: 1.75;
}

/* ── CARDS ── */
.search-panel {
    background: #ffffff;
    border: 2px solid #D6E8F5;
    border-radius: 20px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
}
.section-card {
    background: #ffffff;
    border: 1.5px solid #D6E8F5;
    border-radius: 20px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    color: #1a3a50;
    font-size: 1rem;
    line-height: 1.75;
}
.legend-card {
    background: linear-gradient(135deg, #EBF5FB, #ffffff);
    border: 1.5px solid #AED6F1;
    border-left: 6px solid #2E86C1;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    margin-bottom: 0.9rem;
    color: #173951;
    font-size: 1rem;
}
.metric-card {
    background: #ffffff;
    border: 1.5px solid #D6E8F5;
    border-radius: 20px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
.metric-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1A4B6E;
    line-height: 1.1;
}
.metric-label {
    color: #4a7a99;
    font-size: 0.95rem;
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* ── SERVICE CARDS ── */
.service-card {
    background: #ffffff;
    border: 1.5px solid #D6E8F5;
    border-left: 7px solid #2E86C1;
    border-radius: 20px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    color: #173951;
    transition: box-shadow 0.2s;
}
.service-card:hover {
    box-shadow: 0 8px 24px rgba(46,134,193,0.13);
}
.service-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1A4B6E;
    margin-bottom: 0.25rem;
}
.service-meta {
    color: #4a7a99;
    font-size: 0.97rem;
    margin-bottom: 0.7rem;
    font-weight: 600;
}
.tag {
    display: inline-block;
    background: #EBF5FB;
    color: #1A5276;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    border: 1px solid #AED6F1;
}
.quick-box {
    background: linear-gradient(135deg, #EBF5FB, #F0F9FF);
    border: 1.5px solid #AED6F1;
    border-radius: 18px;
    padding: 1.2rem;
    color: #163a56;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
.assistant-box {
    background: linear-gradient(135deg, #E8F8F2, #F0FBF5);
    border: 1.5px solid #82C9A5;
    border-left: 6px solid #1DB87A;
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    color: #1a4a33;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    margin-top: 1rem;
    margin-bottom: 1rem;
    font-size: 1rem;
    line-height: 1.75;
}
.warning-box {
    background: linear-gradient(135deg, #FFF8EE, #FFFBF5);
    border: 1.5px solid #F0C080;
    border-left: 6px solid #E67E22;
    border-radius: 18px;
    padding: 1rem 1.3rem;
    color: #6E3A0A;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
    font-size: 1rem;
}
.rec-card {
    background: #ffffff;
    border: 1.5px solid #AED6F1;
    border-top: 5px solid #2E86C1;
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 16px rgba(46,134,193,0.08);
    color: #163a56;
    height: 100%;
}
.rec-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: #1A4B6E;
    margin-bottom: 0.25rem;
}
.rec-meta {
    font-size: 0.9rem;
    color: #4a7a99;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.rec-desc {
    font-size: 0.97rem;
    color: #2d4e62;
    line-height: 1.65;
}
.section-note {
    color: #4a7a99;
    font-size: 1rem;
    font-weight: 600;
}
.voice-result {
    background: linear-gradient(135deg, #EBF5FB, #F0F9FF);
    border: 2px dashed #7ABFE8;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    color: #173951;
    margin-top: 0.8rem;
    margin-bottom: 0.9rem;
    font-size: 1rem;
}
.small-muted {
    color: #5c7688;
    font-size: 0.97rem;
}
.star-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}
.stars { color: #F39C12; font-size: 1.2rem; letter-spacing: 2px; }
.star-num { font-weight: 800; color: #1A4B6E; font-size: 0.92rem; }
.star-label { font-size: 0.85rem; color: #4a7a99; font-weight: 600; }
.rating-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 16px;
    margin-top: 0.6rem;
}
.rating-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    color: #2d5a7a;
    font-weight: 600;
}
.rbar-wrap {
    flex: 1;
    background: #D6E8F5;
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    min-width: 40px;
}
.rbar { height: 7px; border-radius: 6px; background: linear-gradient(90deg,#2E86C1,#85C1E9); }

/* ── BUSINESS MODEL specific ── */
.biz-hero {
    background: linear-gradient(135deg, #1A4B6E 0%, #117A8B 100%);
    border-radius: 24px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 12px 32px rgba(26,75,110,0.3);
    position: relative;
    overflow: hidden;
}
.biz-hero::after {
    content: "💼";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.1;
}
.tier-card {
    border-radius: 20px;
    padding: 1.5rem;
    height: 100%;
    box-shadow: 0 4px 18px rgba(0,0,0,0.07);
}
.tier-free { background:#F8FEFF; border:2px solid #AED6F1; }
.tier-pro { background:#EBF5FB; border:3px solid #2E86C1; }
.tier-enterprise { background:#F0FAF5; border:2px solid #82C9A5; }
.tier-price {
    font-size: 2.4rem;
    font-weight: 800;
    color: #1A4B6E;
    line-height: 1;
    margin: 0.5rem 0 1rem;
}
.tier-name {
    font-size: 1.1rem;
    font-weight: 800;
    color: #1A4B6E;
}
.cost-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1rem;
    color: #1a3a50;
}
.cost-table tr {
    border-bottom: 1.5px solid #D6E8F5;
}
.cost-table tr:last-child {
    border-bottom: none;
}
.cost-table td {
    padding: 0.7rem 0.3rem;
    line-height: 1.5;
}
.cost-table .total-row td {
    font-weight: 800;
    font-size: 1.05rem;
    color: #1A4B6E;
    padding-top: 1rem;
    border-top: 2.5px solid #2E86C1;
    border-bottom: none;
}
.step-card {
    background: #ffffff;
    border: 1.5px solid #D6E8F5;
    border-top: 5px solid #2E86C1;
    border-radius: 18px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    height: 100%;
}
.step-num {
    font-size: 2.5rem;
    font-weight: 800;
    color: #D6E8F5;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.step-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #1A4B6E;
    margin-bottom: 0.5rem;
}
.step-desc {
    font-size: 0.95rem;
    color: #2d5a7a;
    line-height: 1.7;
}
.highlight-pill {
    display: inline-block;
    background: linear-gradient(135deg, #2E86C1, #1A5276);
    color: white;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
sidebar_df = st.session_state.services_df.copy()

friendly_categories = ["All"] + sorted([x for x in sidebar_df["FriendlyCategory"].dropna().unique().tolist() if str(x).strip()])
locations = ["All"] + sorted([x for x in sidebar_df["Location"].dropna().unique().tolist() if str(x).strip()])
costs = ["All", "Free", "Partial pay", "Subscription", "Private pay"]
access_modes = ["All"] + sorted([x for x in sidebar_df["AccessMode"].dropna().unique().tolist() if str(x).strip()])

costs = ["All", "Free", "Partial pay", "Subscription", "Private pay"]
access_modes = ["All"] + sorted([x for x in sidebar_df["AccessMode"].dropna().unique().tolist() if str(x).strip()])

st.sidebar.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem;">
    <div style="font-size:1.25rem;font-weight:800;color:#FFD580;letter-spacing:1px;">💙 CCWS Directory</div>
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:2px;">Greater Victoria, BC</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="color:rgba(255,255,255,0.45);font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;padding-left:2px;">Navigate</div>', unsafe_allow_html=True)
nav_items = [("🏠", "Home", 0), ("🔍", "Find Services", 1), ("👤", "My Account", 2), ("🏢", "For Providers", 3), ("💼", "Business Model", 4)]
for icon, label, idx in nav_items:
    active = st.session_state.active_tab == idx
    if st.sidebar.button(f"{icon}  {label}", key=f"nav_{idx}", use_container_width=True):
        st.session_state.active_tab = idx
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="color:rgba(255,255,255,0.45);font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;">Who are you?</div>', unsafe_allow_html=True)
experience_mode = st.sidebar.radio(
    "Who are you?",
    ["Guest / Public", "Senior / Family", "Provider / Subscriber"],
    index=["Guest / Public", "Senior / Family", "Provider / Subscriber"].index(st.session_state.experience_mode),
    label_visibility="collapsed"
)
if experience_mode != st.session_state.experience_mode:
    st.session_state.experience_mode = experience_mode
    if experience_mode == "Senior / Family":
        st.session_state.active_tab = 2
    elif experience_mode == "Provider / Subscriber":
        st.session_state.active_tab = 3
    st.rerun()
st.session_state.experience_mode = experience_mode

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="color:rgba(255,255,255,0.45);font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;">Filters</div>', unsafe_allow_html=True)
selected_category = st.sidebar.selectbox("Service Category", friendly_categories)
selected_location = st.sidebar.selectbox("Location", locations)
selected_cost = st.sidebar.selectbox("Fees", costs)
selected_access_mode = st.sidebar.selectbox("Access Mode", access_modes)

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="color:rgba(255,255,255,0.45);font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;">Quick Search</div>', unsafe_allow_html=True)
engine_query = st.sidebar.text_input("Search services", value=st.session_state.used_engine_query, placeholder="doctor, meals, bus...", label_visibility="collapsed")
sidebar_search_clicked = False
if st.sidebar.button("🔍  Apply & Search", use_container_width=True):
    st.session_state.active_tab = 1
    sidebar_search_clicked = True

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background:rgba(255,215,128,0.07);border:1px solid rgba(255,215,128,0.18);border-radius:12px;padding:0.8rem 0.9rem;text-align:center;">
    <div style="color:#FFD580;font-size:0.8rem;font-weight:800;letter-spacing:0.5px;margin-bottom:4px;">🏆 Built by APEX Consulting</div>
    <div style="color:rgba(255,255,255,0.4);font-size:0.7rem;line-height:1.7;">
        Qian Li · Nilan Beigi<br>Sofiya Golagha · Chaz Alec<br>
        <span style="color:rgba(255,215,128,0.45);">BCIT BITMAN · 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
SENIOR_IMG = "https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=600&q=80"
hero_left, hero_right = st.columns([1.5, 1])
with hero_left:
    st.markdown("""
    <div class="hero-wrap">
        <div style="font-size:0.75rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,0.5);margin-bottom:0.4rem;">Creating Community Wellness Society · Greater Victoria, BC</div>
        <div class="hero-title">💙 Find the Help<br>You Need — Today</div>
        <div class="hero-sub">A free service directory for seniors and caregivers in Greater Victoria. Search by voice, get plain-language results, and connect with the right support.</div>
        <div style="margin-top:1rem;display:flex;gap:0.5rem;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.25rem 0.85rem;font-size:0.82rem;color:white;font-weight:700;">🎙️ Voice Search</span>
            <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.25rem 0.85rem;font-size:0.82rem;color:white;font-weight:700;">🤖 AI Matching</span>
            <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.25rem 0.85rem;font-size:0.82rem;color:white;font-weight:700;">✅ Always Free</span>
            <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.25rem 0.85rem;font-size:0.82rem;color:white;font-weight:700;">📍 35+ Services</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with hero_right:
    st.markdown(f"""
    <div style="border-radius:20px;overflow:hidden;box-shadow:0 10px 28px rgba(0,0,0,0.16);position:relative;">
        <img src="{SENIOR_IMG}" style="width:100%;height:195px;object-fit:cover;display:block;" alt="Senior using tablet"/>
        <div style="position:absolute;bottom:0;left:0;right:0;background:linear-gradient(to top,rgba(26,75,110,0.9),transparent);padding:0.85rem 1rem;">
            <div style="color:white;font-size:0.92rem;font-weight:800;">For seniors, by community.</div>
            <div style="color:rgba(255,255,255,0.72);font-size:0.8rem;">Simple · Clear · Always free</div>
        </div>
    </div>
    <div class="section-card" style="border-top:4px solid #2E86C1;margin-top:0.7rem;padding:0.9rem 1.1rem;">
        <div style="font-size:0.85rem;font-weight:800;color:#1A4B6E;margin-bottom:0.45rem;">📋 Current Session</div>
        <div style="font-size:0.88rem;color:#2d5a7a;line-height:1.9;">
        <strong>Mode:</strong> {st.session_state.experience_mode}<br>
        <strong>User:</strong> {"✅ " + st.session_state.user_name_demo if st.session_state.user_logged_in else "Not logged in"}<br>
        <strong>Provider:</strong> {"✅ " + st.session_state.provider_name_demo if st.session_state.provider_logged_in else "Not logged in"}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Search action helper ----------------
def execute_search(final_query_value, engine_query_value):
    st.session_state.search_query = final_query_value
    st.session_state.final_query = final_query_value
    st.session_state.used_category = selected_category
    st.session_state.used_location = selected_location
    st.session_state.used_cost = selected_cost
    st.session_state.used_access_mode = selected_access_mode
    st.session_state.used_engine_query = engine_query_value
    if has_real_search(final_query_value, selected_category, selected_location, selected_cost, selected_access_mode, engine_query_value):
        results, recommended, reply, summary = assistant_search(
            st.session_state.services_df, final_query_value,
            selected_category, selected_location, selected_cost, selected_access_mode, engine_query_value
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
    execute_search(engine_query.strip(), engine_query.strip())

# APEX footer bar visible on all pages
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(26,75,110,0.07),rgba(17,122,139,0.05));border:1px solid rgba(26,75,110,0.1);border-radius:14px;padding:0.6rem 1.3rem;margin-bottom:0.8rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.4rem;">
    <div style="display:flex;align-items:center;gap:0.5rem;">
        <span style="font-size:1rem;">🏆</span>
        <span style="font-size:0.85rem;font-weight:800;color:#1A4B6E;">APEX Consulting Group</span>
        <span style="font-size:0.8rem;color:#4a7a99;"> · Qian Li · Nilan Beigi · Sofiya Golagha · Chaz Alec</span>
    </div>
    <div style="font-size:0.8rem;color:#4a7a99;">For <strong style="color:#1A4B6E;">CCWS</strong> &nbsp;·&nbsp; BCIT BITMAN 2026 &nbsp;·&nbsp; BSYS-4905</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Tabs (controlled by active_tab) ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "🔍 Find Services",
    "👤 My Account",
    "🏢 For Providers",
    "💼 Business Model"
])

# JS: auto-click the correct tab based on active_tab session state
_active = st.session_state.get("active_tab", 0)
st.markdown(f"""
<script>
(function() {{
    var targetIdx = {_active};
    function clickTab() {{
        var tabButtons = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
        if (tabButtons.length > targetIdx) {{
            tabButtons[targetIdx].click();
        }}
    }}
    setTimeout(clickTab, 120);
}})();
</script>
""", unsafe_allow_html=True)


# ---------------- HOME ----------------
with tab1:
    overview_df = st.session_state.services_df.copy()

    # ── MAIN WELCOME BANNER
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A4B6E 0%,#2E86C1 100%);border-radius:24px;padding:2.2rem 2.4rem;margin-bottom:1.5rem;position:relative;overflow:hidden;box-shadow:0 12px 32px rgba(26,75,110,0.2);">
        <div style="position:absolute;right:-20px;top:-20px;font-size:8rem;opacity:0.07;line-height:1;">💙</div>
        <div style="font-size:0.75rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:rgba(255,255,255,0.5);margin-bottom:0.5rem;">
            Creating Community Wellness Society · Greater Victoria, BC
        </div>
        <div style="font-size:2.1rem;font-weight:800;color:white;line-height:1.25;margin-bottom:0.7rem;">
            Welcome. 👋<br>
            <span style="color:#7EC8E3;">What do you need help with today?</span>
        </div>
        <div style="font-size:1.05rem;color:rgba(255,255,255,0.82);line-height:1.75;max-width:680px;">
            This is a free, easy-to-use directory of community and health services for seniors in Greater Victoria.
            Just tell us what you need — in plain words — and we'll find the right support for you.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── BIG SEARCH BAR ON HOME (sends to Find Services)
    st.markdown('<div style="font-size:1.15rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">🔍 Start here — search for what you need:</div>', unsafe_allow_html=True)
    home_search_col, home_btn_col = st.columns([4, 1])
    with home_search_col:
        home_query = st.text_input(
            "home_search",
            placeholder='e.g.  "I need meals delivered"   or   "help with transportation"   or   "doctor near me"',
            label_visibility="collapsed",
            key="home_search_input"
        )
    with home_btn_col:
        home_search_go = st.button("🔍  Search", use_container_width=True, key="home_search_btn")

    if home_search_go and home_query.strip():
        execute_search(home_query.strip(), "")
        st.session_state.active_tab = 1
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── QUICK-TAP NEED BUTTONS (real search triggers)
    st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.7rem;">Or tap what describes you best:</div>', unsafe_allow_html=True)

    need_items = [
        ("🍽️", "I need food or meals", "food meals delivery"),
        ("🚌", "I need a ride or transport", "transportation bus rides handydart"),
        ("🏥", "I need a doctor or home care", "doctor health home care clinic"),
        ("🏠", "I need housing help", "housing rent shelter"),
        ("🤝", "I want social activities", "social activities community friendly visits"),
        ("💰", "I need financial help", "income support benefits financial"),
        ("♿", "I need mobility support", "mobility walker wheelchair"),
        ("🆘", "I need emergency help", "emergency urgent crisis"),
    ]

    need_cols = st.columns(4)
    for idx, (icon, label, query) in enumerate(need_items):
        with need_cols[idx % 4]:
            if st.button(f"{icon}  {label}", key=f"need_{idx}", use_container_width=True):
                execute_search(query, "")
                st.session_state.active_tab = 1
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── WHAT IS THIS? — plain language for seniors
    st.markdown("""
    <div style="background:#ffffff;border:1.5px solid #D6E8F5;border-left:7px solid #2E86C1;border-radius:20px;padding:1.6rem 1.8rem;margin-bottom:1.2rem;box-shadow:0 4px 16px rgba(0,0,0,0.05);">
        <div style="font-size:1.15rem;font-weight:800;color:#1A4B6E;margin-bottom:0.8rem;">💙 What is this directory?</div>
        <div style="font-size:1rem;color:#2d5a7a;line-height:1.9;">
            Many seniors in Greater Victoria don't know what services are available to them — or how to find them.
            Information is scattered across printed pamphlets, websites, and phone directories that are often out of date.<br><br>
            <strong style="color:#1A4B6E;">This directory changes that.</strong>
            We've collected 35+ verified local services — covering everything from meal delivery and transportation to housing help and medical care —
            and made them searchable in one simple place.<br><br>
            You can search by voice, type a question in plain language, or just tap the type of help you need.
            <strong>It's completely free, and no account is needed to search.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── HOW IT WORKS — 3 steps with real visual
    st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#1A4B6E;margin-bottom:1rem;">How it works — 3 simple steps</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    step_data = [
        ("#2E86C1", "1", "🗣️", "Tell us what you need", "Just type or speak your need in plain words. You don't need to know the official name of any service. \"I need help getting food\" works perfectly."),
        ("#27AE60", "2", "🤖", "Our AI finds the best matches", "We search through 35+ local services instantly and show you the top results — with costs, ratings, location, and contact info all in one place."),
        ("#E67E22", "3", "📞", "Connect with the provider", "Save services you like, send a message, or call directly. If you have an account, providers can follow up with you by phone or email."),
    ]
    for col, (color, num, icon, title, desc) in zip([h1, h2, h3], step_data):
        with col:
            st.markdown(f"""
            <div class="section-card" style="text-align:center;border-top:5px solid {color};">
                <div style="width:56px;height:56px;background:{color};border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 0.8rem;font-size:1.5rem;box-shadow:0 4px 12px rgba(0,0,0,0.15);">{icon}</div>
                <div style="font-size:0.75rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:{color};margin-bottom:0.4rem;">Step {num}</div>
                <div style="font-weight:800;color:#1A4B6E;font-size:1.05rem;margin-bottom:0.6rem;">{title}</div>
                <div style="font-size:0.93rem;color:#4a7a99;line-height:1.75;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── WHAT'S COVERED — category visual grid
    st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.4rem;">What kinds of help are in the directory?</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.95rem;color:#4a7a99;margin-bottom:1rem;">We cover 11 types of support for seniors in Greater Victoria. Click any category to browse.</div>', unsafe_allow_html=True)

    cat_data = [
        ("🏥", "Health & Home Care", "#EBF5FB", "#2E86C1", "Doctors, home care, mental health, dental"),
        ("🍽️", "Food & Nutrition",   "#E8F8F2", "#27AE60", "Meal delivery, food bank, grocery help"),
        ("🚌", "Transportation",     "#FEF9E7", "#E67E22", "Bus, HandyDART, volunteer drivers"),
        ("🏠", "Housing",            "#F5F0FF", "#8E44AD", "Rent help, affordable housing, assisted living"),
        ("🤝", "Social Activities",  "#FFF0F5", "#E74C3C", "Community groups, friendly visits, dementia support"),
        ("💰", "Income Support",     "#F0FFF4", "#16A085", "Benefits, financial aid, energy assistance"),
        ("♿", "Mobility",           "#F0F4FF", "#2980B9", "Wheelchairs, walkers, equipment loans"),
        ("🆘", "Emergency",         "#FFF5F5", "#C0392B", "Crisis lines, legal help, alert systems"),
        ("🏃", "Recreation",        "#F5FFF0", "#27AE60", "Fitness, yoga, aquafit, community activities"),
        ("📚", "Education",         "#FFFBF0", "#F39C12", "Computer literacy, language classes, workshops"),
        ("📋", "Tax & Admin Help",  "#F0F8FF", "#1A5276", "Tax filing, forms, administrative support"),
    ]

    cat_rows = [cat_data[i:i+4] for i in range(0, len(cat_data), 4)]
    for row in cat_rows:
        cols = st.columns(len(row))
        for col, (icon, name, bg, color, desc) in zip(cols, row):
            count = len(overview_df[overview_df["FriendlyCategory"].str.contains(name.split("&")[0].strip(), case=False, na=False)])
            with col:
                st.markdown(f"""
                <div style="background:{bg};border:1.5px solid {color}33;border-top:4px solid {color};border-radius:16px;padding:1rem 0.9rem;text-align:center;cursor:pointer;transition:box-shadow 0.2s;">
                    <div style="font-size:2rem;margin-bottom:0.4rem;">{icon}</div>
                    <div style="font-size:0.92rem;font-weight:800;color:#1A4B6E;line-height:1.3;margin-bottom:0.3rem;">{name}</div>
                    <div style="font-size:0.78rem;color:#4a7a99;line-height:1.4;margin-bottom:0.5rem;">{desc}</div>
                    <div style="font-size:0.78rem;font-weight:700;color:{color};">{count} services</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Browse {name.split('&')[0].strip()}", key=f"cat_browse_{name[:10]}", use_container_width=True):
                    execute_search(name.lower(), "")
                    st.session_state.active_tab = 1
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CCWS + APEX CREDIT at bottom of home
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem;">
        <div style="background:linear-gradient(135deg,#EBF5FB,#F0FFF8);border:1.5px solid #AED6F1;border-radius:18px;padding:1.3rem 1.5rem;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">💙 About CCWS</div>
            <div style="font-size:0.92rem;color:#2d5a7a;line-height:1.8;">
                The <strong>Creating Community Wellness Society</strong> is a non-profit organization in Greater Victoria
                dedicated to improving the health and wellbeing of seniors and their families.
                This directory is one of their initiatives to make community support more accessible.
            </div>
        </div>
        <div style="background:linear-gradient(135deg,#FEF9E7,#FFFBF0);border:1.5px solid #FAD7A0;border-radius:18px;padding:1.3rem 1.5rem;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">🏆 Built by APEX Consulting Group</div>
            <div style="font-size:0.92rem;color:#2d5a7a;line-height:1.8;">
                This platform was designed and built by <strong>Qian Li, Nilan Beigi, Sofiya Golagha, and Chaz Alec</strong>
                from APEX Consulting Group as part of the BCIT BITMAN program, in partnership with CCWS.
                Advisor: Jeff Sawers, BCIT.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FIND SERVICES ----------------
with tab2:

    st.markdown("""
    <div style="text-align:center;padding:0.8rem 0 1.2rem;">
        <div style="font-size:1.8rem;font-weight:800;color:#1A4B6E;margin-bottom:0.4rem;">What do you need help with? \U0001F50D</div>
        <div style="font-size:1rem;color:#4a7a99;">Type your question below, or use your voice. No special words needed — just ask naturally.</div>
    </div>
    """, unsafe_allow_html=True)

    typed_query = st.text_input(
        "Search",
        value=st.session_state.search_query,
        placeholder='e.g.  "I need meals delivered"  or  "help with transportation"  or  "doctor near me"',
        label_visibility="collapsed"
    )
    typed_search_clicked = st.button("\U0001F50D  Search Services", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    voice_col1, voice_col2 = st.columns([2, 1])
    with voice_col1:
        st.markdown("""
        <div class="search-panel" style="padding:1rem 1.3rem;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">\U0001F3A4 Or search with your voice</div>
            <div style="font-size:0.9rem;color:#4a7a99;">Press the button, speak your question, then click Search with Voice.</div>
        </div>
        """, unsafe_allow_html=True)
        if VOICE_AVAILABLE:
            voice_text_input = speech_to_text(
                language="en",
                start_prompt="\U0001F3A4  Start Speaking",
                stop_prompt="\u23F9  Stop",
                just_once=True,
                use_container_width=True,
                key="main_voice_search",
            )
            if voice_text_input:
                st.session_state.voice_text_display = voice_text_input
        else:
            st.info("Voice package not installed.")
        if st.session_state.voice_text_display:
            st.markdown('<div class="voice-result">\U0001F3A4 <strong>You said:</strong> &nbsp;' + str(st.session_state.voice_text_display) + '</div>', unsafe_allow_html=True)

    with voice_col2:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        voice_search_clicked = st.button("\U0001F3A4  Search with Voice", use_container_width=True)
        reset_clicked = st.button("\u21BA  Clear & Start Over", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.6rem;">Or tap a common need:</div>', unsafe_allow_html=True)
    shortcut_cols = st.columns(3)
    shortcut_items = list(PROFILE_SHORTCUTS.items())
    for i, (label, query_value) in enumerate(shortcut_items):
        with shortcut_cols[i % 3]:
            if st.button(label, key=f"shortcut_{i}", use_container_width=True):
                execute_search(query_value, "")

    if typed_search_clicked:
        execute_search(typed_query.strip(), "")
    if voice_search_clicked:
        execute_search(st.session_state.voice_text_display.strip(), "")
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
        st.session_state.used_access_mode = "All"
        st.session_state.used_engine_query = ""
        st.session_state.voice_text_display = ""
        st.rerun()

    if st.session_state.searched and st.session_state.final_query:
        st.markdown(
            '<div class="assistant-box" style="margin-top:1.5rem;">' +
            '<div style="font-size:0.9rem;color:#4a7a99;margin-bottom:0.3rem;">You searched for:</div>' +
            f'<div style="font-size:1.1rem;font-weight:800;color:#1A4B6E;">"{st.session_state.final_query}"</div>' +
            f'<div style="font-size:0.9rem;color:#27AE60;margin-top:0.3rem;">Detected need: {st.session_state.assistant_summary}</div>' +
            '</div>',
            unsafe_allow_html=True
        )

    if st.session_state.assistant_reply:
        st.markdown(
            '<div class="assistant-box">' +
            '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">' +
            '<span style="font-size:1.3rem;">\U0001F916</span>' +
            '<span style="font-weight:800;color:#1A4B6E;font-size:1rem;">AI Assistant</span></div>' +
            str(st.session_state.assistant_reply) +
            '</div>',
            unsafe_allow_html=True
        )

    if not st.session_state.recommended.empty:
        st.markdown('<div style="font-size:1.2rem;font-weight:800;color:#1A4B6E;margin:1rem 0 0.8rem;">\u2B50 Best matches for you</div>', unsafe_allow_html=True)
        rec_cols = st.columns(min(3, len(st.session_state.recommended)))
        for i, (_, row) in enumerate(st.session_state.recommended.iterrows()):
            with rec_cols[i]:
                _rating = rating_block_html(row)
                _verif = verification_label(row)
                _subcat = row['Subcategory'] if str(row['Subcategory']).strip() else 'General service'
                _fee = row.get('FeeType', row.get('CostType', ''))
                st.markdown(
                    '<div class="rec-card">' +
                    f'<div class="rec-title">{row["ServiceName"]}</div>' +
                    f'<div class="rec-meta">{row["FriendlyCategory"]} &nbsp;\u00B7&nbsp; {row["Location"]} &nbsp;\u00B7&nbsp; <strong style="color:#27AE60">{_fee}</strong></div>' +
                    f'<div class="rec-desc">{row["Description"]}</div>' +
                    _rating +
                    f'<div style="margin-top:0.6rem;"><span class="tag">{_subcat}</span><span class="tag">{_verif}</span></div>' +
                    '</div>',
                    unsafe_allow_html=True
                )

    results = st.session_state.results
    if not results.empty:
        st.markdown(f'<div class="section-note" style="margin:1rem 0 0.5rem;">Found <strong>{len(results)}</strong> services matching your search</div>', unsafe_allow_html=True)

        for _, row in results.iterrows():
            _rating = rating_block_html(row)
            _verif = verification_label(row)
            _subcat = row['Subcategory'] if str(row['Subcategory']).strip() else 'General'
            st.markdown(
                '<div class="service-card">' +
                f'<div class="service-title">{row["ServiceName"]}</div>' +
                f'<div class="service-meta">{row["FriendlyCategory"]} &nbsp;\u00B7&nbsp; {row["AreaOfVictoria"]} &nbsp;\u00B7&nbsp; <strong style="color:#27AE60">{row["FeeType"]}</strong> &nbsp;\u00B7&nbsp; {row["AccessMode"]}</div>' +
                f'<div style="margin-bottom:0.6rem;font-size:1rem;color:#2d5a7a;line-height:1.7;">{row["Description"]}</div>' +
                _rating +
                f'<div style="margin-top:0.7rem;"><span class="tag">\U0001F4C2 {_subcat}</span><span class="tag">\U0001F3E2 {row["Organization"]}</span><span class="tag">\U0001F68C Transit: {row["TransitAccess"]}</span></div>' +
                f'<div style="margin-top:0.6rem;color:#4a7a99;font-size:0.88rem;">{_verif}</div>' +
                '</div>',
                unsafe_allow_html=True
            )

            action_col1, action_col2 = st.columns([1, 1])
            with action_col1:
                if st.session_state.user_logged_in:
                    if st.button(f"\U0001F4BE Save This Service", key=f"save_{row['ServiceID']}", use_container_width=True):
                        save_service(row["ServiceID"])
                        st.success(f"\u2705 {row['ServiceName']} saved to My Account!")
                else:
                    st.caption("\U0001F464 Log in to My Account to save services")
            with action_col2:
                if st.session_state.user_logged_in:
                    with st.expander(f"\U0001F4DE Contact {row['ServiceName']}"):
                        request_message = st.text_area("Your message", placeholder="Hello, I would like more information.", key=f"msg_{row['ServiceID']}")
                        request_contact = st.selectbox("How should they contact you?", ["Phone", "Email"], key=f"contact_pref_{row['ServiceID']}")
                        if st.button("\U0001F4E8 Send Message", key=f"send_req_{row['ServiceID']}", use_container_width=True):
                            add_user_request(row, request_message.strip() or "Requesting more information.", request_contact)
                            st.success("\u2705 Message sent! The provider will contact you soon.")
                else:
                    st.caption("\U0001F464 Log in to My Account to contact providers")

        st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#1A4B6E;margin:1.2rem 0 0.5rem;">\U0001F4CB Full details for a service</div>', unsafe_allow_html=True)
        selected_service = st.selectbox("Choose a service to see full details", results["ServiceName"].tolist(), label_visibility="collapsed")
        row = results[results["ServiceName"] == selected_service].iloc[0]
        left_d, right_d = st.columns([1.5, 1])
        with left_d:
            st.markdown(f"### {row['ServiceName']}")
            st.write(row["Description"])
            st.markdown(f"**Type of service:** {row['FriendlyCategory']}")
            st.markdown(f"**Location:** {row['Location']}")
            st.markdown(f"**Who can use it:** {row['Eligibility']}")
            st.markdown(f"**How to access:** {row['AccessMode']}")
            st.markdown("#### \u2B50 User Experience Ratings")
            st.markdown(rating_block_html(row), unsafe_allow_html=True)
        with right_d:
            st.markdown('<div class="quick-box"><div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.8rem;">\U0001F4DE Quick Contact Info</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Cost:** {row['FeeType']}")
            st.markdown(f"**Organization:** {row['Organization']}")
            st.markdown(f"**Phone:** {row['Phone'] or 'Not listed'}")
            st.markdown(f"**Email:** {row['Email'] or 'Not listed'}")
            st.markdown(f"**Website:** {row['Website'] or 'Not listed'}")
            st.markdown(f"**Transit access:** {row['TransitAccess']}")
            st.markdown(f"**Verification:** {verification_label(row)}")

    elif st.session_state.searched:
        st.markdown('<div class="warning-box" style="text-align:center;margin-top:1.5rem;"><div style="font-size:1.1rem;font-weight:800;margin-bottom:0.5rem;">No services found for that search \U0001F614</div><div style="font-size:0.97rem;line-height:1.7;">Try simpler words like: <strong>doctor, meals, bus, housing, food, help at home</strong></div></div>', unsafe_allow_html=True)

    if not st.session_state.searched:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.8rem;">\U0001F4AC Not sure what to search? Here is a guide to common service types:</div>', unsafe_allow_html=True)
        lg1, lg2 = st.columns(2)
        with lg1:
            for title, desc in USER_FRIENDLY_LEGEND[:6]:
                st.markdown(f'<div class="legend-card"><strong>{title}</strong><br><span style="color:#4a7a99;font-size:0.93rem;">{desc}</span></div>', unsafe_allow_html=True)
        with lg2:
            for title, desc in USER_FRIENDLY_LEGEND[6:]:
                st.markdown(f'<div class="legend-card"><strong>{title}</strong><br><span style="color:#4a7a99;font-size:0.93rem;">{desc}</span></div>', unsafe_allow_html=True)

# ---------------- MY ACCOUNT (Senior Portal) ----------------
with tab3:

    if not st.session_state.user_logged_in:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 1.5rem;">
            <div style="font-size:2rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">👤 My Account</div>
            <div style="font-size:1rem;color:#4a7a99;max-width:600px;margin:0 auto;line-height:1.8;">
                Create your personal account to save services you like, build a shortlist, and send messages to providers — all in one place.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.markdown("""
            <div class="section-card" style="border-top:5px solid #2E86C1;text-align:center;padding:2rem 1.5rem;">
                <div style="font-size:2.5rem;margin-bottom:0.8rem;">👋</div>
                <div style="font-size:1.2rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">Welcome! Just enter your name to get started.</div>
                <div style="font-size:0.93rem;color:#4a7a99;margin-bottom:1.2rem;">No password needed. No personal health information collected.</div>
            </div>
            """, unsafe_allow_html=True)
            user_name = st.text_input("Your name", value=st.session_state.user_name_demo, placeholder="e.g. Margaret or John's Family")
            user_role = st.selectbox("I am a:", ["Senior", "Family Member", "Caregiver"])
            if st.button("✅  Create My Account & Sign In", use_container_width=True):
                if user_name.strip():
                    st.session_state.user_logged_in = True
                    st.session_state.user_name_demo = user_name.strip()
                    st.session_state.user_role_demo = user_role
                    st.session_state.user_profile["full_name"] = user_name.strip()
                    st.session_state.user_profile["role"] = user_role
                    st.rerun()
                else:
                    st.warning("Please enter your name to continue.")

        st.markdown("<br>", unsafe_allow_html=True)
        feat1, feat2, feat3 = st.columns(3)
        with feat1:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">💾</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Save Services</div><div style="font-size:0.9rem;color:#4a7a99;">Keep a shortlist of services you want to remember or come back to.</div></div>""", unsafe_allow_html=True)
        with feat2:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">📨</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Message Providers</div><div style="font-size:0.9rem;color:#4a7a99;">Send a message directly to a service provider and they'll get back to you.</div></div>""", unsafe_allow_html=True)
        with feat3:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">👤</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Your Profile</div><div style="font-size:0.9rem;color:#4a7a99;">Set your preferences so we can find the most relevant services for you.</div></div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="assistant-box" style="display:flex;align-items:center;gap:1rem;">
            <div style="font-size:2.5rem;">👋</div>
            <div>
                <div style="font-size:1.2rem;font-weight:800;color:#1A4B6E;">Welcome back, {st.session_state.user_profile["full_name"]}!</div>
                <div style="font-size:0.95rem;color:#4a7a99;">Logged in as: {st.session_state.user_profile["role"]} &nbsp;·&nbsp; Greater Victoria</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_tab1, user_tab2, user_tab3, user_tab4 = st.tabs([
            "👤 My Profile", "💾 Saved Services", "📨 My Messages", "📋 My Needs Summary"
        ])

        with user_tab1:
            st.markdown("### My Profile & Preferences")
            st.markdown("<div style='color:#4a7a99;font-size:0.95rem;margin-bottom:1rem;'>Fill in your preferences so we can find the most relevant services for you.</div>", unsafe_allow_html=True)
            with st.form("user_profile_form"):
                up1, up2 = st.columns(2)
                with up1:
                    full_name = st.text_input("Your name", value=st.session_state.user_profile["full_name"])
                    role = st.selectbox(
                        "I am a:",
                        ["Senior", "Family Member", "Caregiver"],
                        index=["Senior", "Family Member", "Caregiver"].index(st.session_state.user_profile["role"])
                    )
                    location = st.text_input("My area in Victoria", value=st.session_state.user_profile["location"])
                    preferred_contact = st.selectbox(
                        "Best way to reach me:",
                        ["Phone", "Email"],
                        index=["Phone", "Email"].index(st.session_state.user_profile["preferred_contact"])
                    )
                with up2:
                    cost_preference = st.selectbox(
                        "Cost preference:",
                        ["Public", "Private", "Shared"],
                        index=["Public", "Private", "Shared"].index(st.session_state.user_profile["cost_preference"])
                    )
                    mobility_needs = st.text_input("Mobility or accessibility needs (optional)", value=st.session_state.user_profile["mobility_needs"])
                    transportation_needs = st.text_input("Transportation needs (optional)", value=st.session_state.user_profile["transportation_needs"])
                    support_interests = st.multiselect(
                        "What kinds of support are you looking for?",
                        ["Doctor / Clinic", "Food & Nutrition", "Housing", "Transportation", "Community Support", "Financial Help", "Safety & Protection"],
                        default=st.session_state.user_profile["support_interests"]
                    )
                notes = st.text_area(
                    "Any other notes (optional)",
                    value=st.session_state.user_profile["notes"],
                    placeholder="General preferences only, not medical records."
                )
                profile_save = st.form_submit_button("Save Profile", use_container_width=True)

                if profile_save:
                    st.session_state.user_profile = {
                        "full_name": full_name,
                        "role": role,
                        "location": location,
                        "preferred_contact": preferred_contact,
                        "cost_preference": cost_preference,
                        "mobility_needs": mobility_needs,
                        "transportation_needs": transportation_needs,
                        "support_interests": support_interests,
                        "notes": notes,
                    }
                    st.success("Profile saved.")

        with user_tab2:
            st.markdown("### Saved Services")
            saved_ids = st.session_state.saved_services
            if not saved_ids:
                st.info("No saved services yet. Save services from the Search tab.")
            else:
                saved_df = st.session_state.services_df[st.session_state.services_df["ServiceID"].isin(saved_ids)].copy()
                for _, row in saved_df.iterrows():
                    st.markdown(f"""
                    <div class="service-card">
                        <div class="service-title">{row['ServiceName']}</div>
                        <div class="service-meta">{row['FriendlyCategory']} • {row['Location']} • {row.get('FeeType', row.get('CostType', ''))}</div>
                        <div style="margin-bottom:0.8rem;">{row['Description']}</div>
                        <span class="tag">Provider: {row['Provider']}</span>
                        <span class="tag">Access: {row['AccessMode']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Remove from Saved", key=f"remove_saved_{row['ServiceID']}", use_container_width=True):
                        remove_saved_service(row["ServiceID"])
                        st.rerun()

        with user_tab3:
            st.markdown("### My Requests")
            user_requests_df = st.session_state.user_requests.copy()
            user_requests_df = user_requests_df[user_requests_df["UserName"] == st.session_state.user_profile["full_name"]]

            if user_requests_df.empty:
                st.info("No requests sent yet.")
            else:
                for _, req in user_requests_df.iterrows():
                    st.markdown(f"""
                    <div class="section-card">
                        <strong>{req['ServiceName']}</strong><br>
                        <span class="tag">Provider: {req['Provider']}</span>
                        <span class="tag">Status: {req['Status']}</span>
                        <span class="tag">Contact: {req['PreferredContact']}</span>
                        <p style="margin-top:0.6rem;">{req['Message']}</p>
                        <small>Sent: {req['CreatedAt']}</small>
                    </div>
                    """, unsafe_allow_html=True)

        with user_tab4:
            st.markdown("### My Needs Summary")
            profile = st.session_state.user_profile
            interests = ", ".join(profile["support_interests"]) if profile["support_interests"] else "not set yet"
            summary_text = (
                f"{profile['full_name'] or 'This user'} is looking for support in {profile['location']}. "
                f"Role: {profile['role']}. Preferred contact: {profile['preferred_contact']}. "
                f"Cost preference: {profile['cost_preference']}. "
                f"Support interests: {interests}. "
            )
            if profile["mobility_needs"]:
                summary_text += f"Mobility / accessibility needs: {profile['mobility_needs']}. "
            if profile["transportation_needs"]:
                summary_text += f"Transportation needs: {profile['transportation_needs']}. "
            if profile["notes"]:
                summary_text += f"Notes: {profile['notes']}."

            st.markdown(f"""
            <div class="section-card">
                <strong>Profile summary</strong><br><br>
                {summary_text}
            </div>
            """, unsafe_allow_html=True)

            if profile["support_interests"]:
                recommended_from_profile = st.session_state.services_df[
                    st.session_state.services_df["FriendlyCategory"].isin(profile["support_interests"])
                ].head(5)
                if not recommended_from_profile.empty:
                    st.markdown("### Suggested services based on your profile")
                    for _, row in recommended_from_profile.iterrows():
                        st.markdown(f"""
                        <div class="service-card">
                            <div class="service-title">{row['ServiceName']}</div>
                            <div class="service-meta">{row['FriendlyCategory']} • {row['Location']} • {row.get('FeeType', row.get('CostType', ''))}</div>
                            <div>{row['Description']}</div>
                        </div>
                        """, unsafe_allow_html=True)

        if st.button("Log Out of Senior / Family Portal", use_container_width=True):
            st.session_state.user_logged_in = False
            st.session_state.user_name_demo = ""
            st.session_state.user_role_demo = "Senior"
            st.rerun()

# ---------------- FOR PROVIDERS ----------------
with tab4:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.2rem;">
        <div style="font-size:1.8rem;font-weight:800;color:#1A4B6E;margin-bottom:0.4rem;">🏢 Provider Portal</div>
        <div style="font-size:1rem;color:#4a7a99;">Manage your service listings and respond to senior contact requests.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        This is a polished provider workflow demo. Providers can manage service entries
        and view incoming user contact requests.
    </div>
    """, unsafe_allow_html=True)

    login_left, login_right = st.columns([1, 1])
    with login_left:
        provider_name = st.text_input(
            "Provider organization name",
            value=st.session_state.provider_name_demo,
            placeholder="Example: Tall Tree Health"
        )
    with login_right:
        if st.button("Demo Login", use_container_width=True):
            if provider_name.strip():
                st.session_state.provider_logged_in = True
                st.session_state.provider_name_demo = provider_name.strip()
                st.rerun()

    if st.session_state.provider_logged_in:
        st.markdown(f"""
        <div class="assistant-box">
            <strong>Logged in as:</strong> {st.session_state.provider_name_demo}<br>
            <strong>Subscription status:</strong> Active (demo)
        </div>
        """, unsafe_allow_html=True)

        provider_subtab1, provider_subtab2 = st.tabs(["Manage Services", "Provider Inbox"])

        with provider_subtab1:
            st.markdown("### Add a new service entry")

            with st.form("provider_add_service_form"):
                p1, p2 = st.columns(2)

                with p1:
                    service_name = st.text_input("Service Name")
                    organization = st.text_input(
                        "Organization Name",
                        value=st.session_state.provider_name_demo
                    )
                    contact_person = st.text_input("Contact Person")

                    category = st.selectbox(
                        "Category",
                        [
                            "Housing",
                            "Income Support",
                            "Health & Home Care",
                            "Emergency",
                            "Food & Nutrition",
                            "Social Activities",
                            "Taxes & Administrative Support",
                            "Mobility",
                            "Transportation",
                            "Recreation",
                            "Education"
                        ]
                    )

                    subcategory = st.text_input("Subcategory")
                    eligibility = st.text_input("Eligibility", placeholder="Who is this service for?")

                    p_location = st.text_input("Location", value="Victoria")
                    area_of_victoria = st.text_input("Area of Victoria")

                with p2:
                    transit_access = st.selectbox("Transit Access", ["Yes", "No", "Limited", "Unknown"])

                    access_mode = st.selectbox(
                        "Access Mode",
                        ["In person", "Online", "Both"]
                    )

                    website = st.text_input("Website", placeholder="https://...")
                    phone = st.text_input("Phone")
                    email = st.text_input("Email")

                    communication_options = st.multiselect(
                        "Communication Options",
                        ["Phone", "Email", "Chat", "FAQ"]
                    )
                    chat_available = "Yes" if "Chat" in communication_options else "No"
                    faq_available = "Yes" if "FAQ" in communication_options else "No"

                    fee_type = st.selectbox(
                        "Fees",
                        ["Free", "Partial Pay", "Subscription", "Private Pay"]
                    )

                description = st.text_area(
                    "Description",
                    placeholder="Plain-language description of the service"
                )

                st.markdown("### User Experience Rating")

                rating = st.slider("Overall Rating", 0, 5, 3)
                friendly_score = st.slider("Friendly", 0, 5, 3)
                efficient_score = st.slider("Efficient", 0, 5, 3)
                easy_to_understand_score = st.slider("Easy to Understand", 0, 5, 3)
                accessibility_score = st.slider("Accessibility", 0, 5, 3)

                submitted = st.form_submit_button("Save Service Entry", use_container_width=True)

                if submitted:
                    if service_name.strip() and description.strip():
                        add_provider_service(
                            service_name=service_name.strip(),
                            category=category,
                            subcategory=subcategory.strip(),
                            organization=organization.strip() or st.session_state.provider_name_demo,
                            contact_person=contact_person.strip(),
                            location=p_location.strip() or "Victoria",
                            area_of_victoria=area_of_victoria.strip() or "Victoria",
                            transit_access=transit_access,
                            access_mode=access_mode,
                            website=website.strip(),
                            phone=phone.strip(),
                            email=email.strip(),
                            communication_options=communication_options,
                            fee_type=fee_type,
                            description=description.strip(),
                            rating=rating,
                            friendly=friendly_score,
                            efficient=efficient_score,
                            easy_to_understand=easy_to_understand_score,
                            accessibility=accessibility_score,
                        )
                        st.success("Service entry added to the demo.")
                        st.rerun()
                    else:
                        st.warning("Service name and description are required.")

        with provider_subtab2:
            st.markdown("### Incoming contact requests")
            provider_requests = st.session_state.user_requests.copy()
            provider_requests = provider_requests[provider_requests["Provider"].str.lower() == st.session_state.provider_name_demo.lower()]

            if provider_requests.empty:
                st.info("No incoming requests yet.")
            else:
                for _, req in provider_requests.iterrows():
                    st.markdown(f"""
                    <div class="section-card">
                        <strong>{req['ServiceName']}</strong><br>
                        <span class="tag">From: {req['UserName']}</span>
                        <span class="tag">Role: {req['UserRole']}</span>
                        <span class="tag">Preferred contact: {req['PreferredContact']}</span>
                        <span class="tag">Status: {req['Status']}</span>
                        <p style="margin-top:0.6rem;">{req['Message']}</p>
                        <small>Received: {req['CreatedAt']}</small>
                    </div>
                    """, unsafe_allow_html=True)

                    new_status = st.selectbox(
                        "Update status",
                        ["Pending", "Replied", "Closed"],
                        index=["Pending", "Replied", "Closed"].index(req["Status"]),
                        key=f"status_{req['RequestID']}"
                    )
                    if st.button("Save Status", key=f"save_status_{req['RequestID']}", use_container_width=True):
                        st.session_state.user_requests.loc[
                            st.session_state.user_requests["RequestID"] == req["RequestID"], "Status"
                        ] = new_status
                        st.success("Request status updated.")
                        st.rerun()

        if st.button("Log Out of Provider Portal", use_container_width=True):
            st.session_state.provider_logged_in = False
            st.session_state.provider_name_demo = ""
            st.rerun()
    else:
        st.info("Use Demo Login to view the provider/subscriber workflow.")

# ---------------- Business Model & Financial Viability ----------------
with tab5:

    # HERO
    st.markdown("""
    <div class="biz-hero">
        <div style="font-size:0.82rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,0.55);margin-bottom:0.5rem;">APEX Consulting Group · BCIT BITMAN 2026 · For CCWS</div>
        <div style="font-size:2rem;font-weight:800;color:#ffffff;line-height:1.2;margin-bottom:0.6rem;">
            Stop Managing Spreadsheets.<br>Start Getting Found.
        </div>
        <div style="font-size:1.05rem;color:rgba(255,255,255,0.85);line-height:1.75;max-width:800px;">
            This page is for <strong style="color:#FFD580;">service providers</strong> — clinics, non-profits, health authorities —
            who want seniors to find their services easily. Here's why switching to this platform makes sense, and what it costs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CLEAR ROLE EXPLANATION
    st.markdown("### 🧭 Who Does What? Let's Be Clear.")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class="section-card" style="border-top:5px solid #2E86C1;text-align:center;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">🏛️</div>
            <div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">CCWS</div>
            <div style="font-size:0.88rem;font-weight:700;color:#2E86C1;margin-bottom:0.8rem;">OWNS THE PLATFORM</div>
            <div style="font-size:0.95rem;color:#2d5a7a;line-height:1.8;">
                Creating Community Wellness Society built and owns this directory.
                They are responsible for keeping it running, verified, and up to date.
                Think of them as the <strong>landlord of the building</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="section-card" style="border-top:5px solid #27AE60;text-align:center;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">🏥</div>
            <div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">You — The Provider</div>
            <div style="font-size:0.88rem;font-weight:700;color:#27AE60;margin-bottom:0.8rem;">LISTS SERVICES HERE</div>
            <div style="font-size:0.95rem;color:#2d5a7a;line-height:1.8;">
                Your clinic, non-profit, or health authority pays a small monthly fee to
                list your services, keep them updated, and receive contact requests from seniors.
                Think of it as <strong>renting a storefront</strong> where seniors already look.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="section-card" style="border-top:5px solid #F39C12;text-align:center;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">👴</div>
            <div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">Seniors & Families</div>
            <div style="font-size:0.88rem;font-weight:700;color:#E67E22;margin-bottom:0.8rem;">ALWAYS FREE</div>
            <div style="font-size:0.95rem;color:#2d5a7a;line-height:1.8;">
                Seniors and caregivers search, save services, and contact providers
                completely for free — always. No sign-up required to browse.
                The platform exists <strong>for them</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # THE PROBLEM PROVIDERS HAVE RIGHT NOW
    st.markdown("### 😩 The Problem Providers Have Right Now")
    st.markdown("""
    <div class="section-card" style="border-left:7px solid #E74C3C;">
        <div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:1rem;">If you run a clinic, non-profit, or community program — you probably recognize this:</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div style="background:#FEF9E7;border:1.5px solid #F9E79F;border-radius:12px;padding:1rem;">
                <div style="font-weight:800;color:#7D6608;margin-bottom:0.4rem;">📋 Your info is scattered</div>
                <div style="font-size:0.93rem;color:#7D6608;line-height:1.7;">Your service info lives in a PDF, a Facebook post, your website footer, and a printed flyer from 2021 — none of them match.</div>
            </div>
            <div style="background:#FEF9E7;border:1.5px solid #F9E79F;border-radius:12px;padding:1rem;">
                <div style="font-weight:800;color:#7D6608;margin-bottom:0.4rem;">📞 Seniors can't find you</div>
                <div style="font-size:0.93rem;color:#7D6608;line-height:1.7;">Seniors call the wrong number, show up on the wrong day, or give up entirely because finding your service is too confusing.</div>
            </div>
            <div style="background:#FEF9E7;border:1.5px solid #F9E79F;border-radius:12px;padding:1rem;">
                <div style="font-weight:800;color:#7D6608;margin-bottom:0.4rem;">🔄 You update the same info in 5 places</div>
                <div style="font-size:0.93rem;color:#7D6608;line-height:1.7;">Every time your hours change or a program ends, someone has to email 3 directories, update the website, and post on social media.</div>
            </div>
            <div style="background:#FEF9E7;border:1.5px solid #F9E79F;border-radius:12px;padding:1rem;">
                <div style="font-weight:800;color:#7D6608;margin-bottom:0.4rem;">📊 You don't know who needs what</div>
                <div style="font-size:0.93rem;color:#7D6608;line-height:1.7;">You have no idea which services seniors search for most, what questions they ask, or which gaps in your community are going unmet.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # THE SOLUTION
    st.markdown("### ✅ What This Platform Does Instead")
    s1, s2 = st.columns([1, 1])
    with s1:
        st.markdown("""
        <div class="section-card" style="border-left:7px solid #27AE60;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:1rem;">As a provider subscriber, you get one place to:</div>
            <div style="font-size:0.97rem;color:#2d5a7a;line-height:2.2;">
                ✅ &nbsp;<strong>List all your services</strong> — unlimited, always current<br>
                ✅ &nbsp;<strong>Update info in 30 seconds</strong> — change hours, fees, eligibility anytime<br>
                ✅ &nbsp;<strong>Receive contact requests</strong> — seniors message you directly through the app<br>
                ✅ &nbsp;<strong>Show up in AI search</strong> — when a senior says "I need meals delivered," your service appears<br>
                ✅ &nbsp;<strong>See your ratings</strong> — know what seniors think about your service<br>
                ✅ &nbsp;<strong>Priority placement</strong> — subscribers appear before unlisted organizations
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="section-card" style="border-left:7px solid #2E86C1;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.8rem;">🎯 Real Example</div>
            <div style="background:#EBF5FB;border-radius:12px;padding:1rem 1.1rem;margin-bottom:0.8rem;">
                <div style="font-size:0.9rem;font-weight:700;color:#1A4B6E;margin-bottom:0.3rem;">Old way (without this platform):</div>
                <div style="font-size:0.9rem;color:#2d5a7a;line-height:1.7;">
                    Margaret, 74, Googles "meal delivery Victoria BC." She finds 3 websites with conflicting info, calls one wrong number, gives up.
                    Your Meals on Wheels program missed a client.
                </div>
            </div>
            <div style="background:#E8F8F2;border-radius:12px;padding:1rem 1.1rem;">
                <div style="font-size:0.9rem;font-weight:700;color:#1A5632;margin-bottom:0.3rem;">New way (with this platform):</div>
                <div style="font-size:0.9rem;color:#2d5a7a;line-height:1.7;">
                    Margaret says "I need food help" into her phone. The AI finds your program instantly.
                    She clicks "Contact Provider." You receive her message. Done.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PRICING - simple
    st.markdown("### 💰 Simple, Transparent Pricing")
    st.markdown("<div style='color:#4a7a99;font-size:1rem;margin-bottom:1.2rem;font-weight:600;'>No setup fees. No long contracts. Cancel anytime.</div>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class="tier-card tier-free">
            <div class="tier-name">🔓 Free Listing</div>
            <div class="tier-price">$0<span style="font-size:1rem;font-weight:400;color:#4a7a99;">/month</span></div>
            <hr style="border:1.5px solid #D6E8F5;margin:0.8rem 0;">
            <ul style="color:#2d5a7a;font-size:0.97rem;line-height:2.2;padding-left:1.1rem;">
                <li>List up to 3 services</li>
                <li>Basic contact info shown</li>
                <li>Appear in search results</li>
            </ul>
            <div style="margin-top:1rem;padding:0.75rem;background:#EBF5FB;border-radius:10px;font-size:0.88rem;color:#1A4B6E;font-weight:600;">
                👥 Good for small volunteer groups
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="tier-card tier-pro" style="position:relative;">
            <div style="position:absolute;top:-14px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg,#2E86C1,#1A5276);color:white;font-size:0.75rem;font-weight:800;padding:4px 18px;border-radius:999px;letter-spacing:1px;white-space:nowrap;">⭐ RECOMMENDED</div>
            <div style="margin-top:0.5rem;">
            <div class="tier-name">🏥 Provider Subscriber</div>
            <div class="tier-price">$49–$99<span style="font-size:1rem;font-weight:400;color:#4a7a99;">/month</span></div>
            <hr style="border:1.5px solid #AED6F1;margin:0.8rem 0;">
            <ul style="color:#2d5a7a;font-size:0.97rem;line-height:2.2;padding-left:1.1rem;">
                <li><strong>Unlimited</strong> service listings</li>
                <li>Update info <strong>anytime</strong></li>
                <li>Receive senior <strong>contact requests</strong></li>
                <li><strong>Priority</strong> in search results</li>
                <li>See ratings &amp; feedback</li>
                <li>Provider inbox &amp; response tools</li>
            </ul>
            <div style="margin-top:1rem;padding:0.75rem;background:#D6EAF8;border-radius:10px;font-size:0.88rem;color:#1A4B6E;font-weight:600;">
                🏥 For clinics, health authorities, non-profits
            </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="tier-card tier-enterprise">
            <div class="tier-name">🏛️ Regional License</div>
            <div class="tier-price">Custom<span style="font-size:1rem;font-weight:400;color:#4a7a99;">/year</span></div>
            <hr style="border:1.5px solid #A9DFBF;margin:0.8rem 0;">
            <ul style="color:#2d5a7a;font-size:0.97rem;line-height:2.2;padding-left:1.1rem;">
                <li><strong>Your logo</strong>, your branding</li>
                <li>Deploy for any region of BC</li>
                <li>Full analytics dashboard</li>
                <li>API integration with your systems</li>
                <li>Dedicated onboarding support</li>
            </ul>
            <div style="margin-top:1rem;padding:0.75rem;background:#E8F8F2;border-radius:10px;font-size:0.88rem;color:#1A5632;font-weight:600;">
                🏛️ For municipalities &amp; health regions
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # THE MATH - simple
    st.markdown("### 🔢 The Numbers — Simple as That")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="section-card" style="border-top:5px solid #E74C3C;">
            <div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:1rem;">🔨 Cost to build the full platform</div>
            <table class="cost-table">
                <tr><td>Build the app</td><td style="text-align:right;font-weight:700;">$16,000 – $30,000</td></tr>
                <tr><td>Load 200+ verified services</td><td style="text-align:right;font-weight:700;">$2,000 – $4,000</td></tr>
                <tr><td>AI &amp; voice search</td><td style="text-align:right;font-weight:700;">$3,000 – $6,000</td></tr>
                <tr><td>Design &amp; accessibility</td><td style="text-align:right;font-weight:700;">$2,000 – $4,000</td></tr>
                <tr class="total-row"><td>Total build cost</td><td style="text-align:right;">~$23,000 – $44,000</td></tr>
            </table>
            <div style="margin-top:0.9rem;padding:0.85rem 1rem;background:#FEF9E7;border:1.5px solid #F9E79F;border-radius:12px;font-size:0.9rem;color:#7D6608;line-height:1.6;">
                💡 <strong>The demo you're looking at right now</strong> was built in 10 weeks by 4 students at near-zero cost. The hard part is done.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="section-card" style="border-top:5px solid #27AE60;">
            <div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:1rem;">📅 Cost to run it per year</div>
            <table class="cost-table">
                <tr><td>Cloud hosting</td><td style="text-align:right;font-weight:700;">$600 – $1,200</td></tr>
                <tr><td>AI &amp; search</td><td style="text-align:right;font-weight:700;">$300 – $900</td></tr>
                <tr><td>Keeping data current</td><td style="text-align:right;font-weight:700;">$1,500 – $3,000</td></tr>
                <tr><td>Part-time admin</td><td style="text-align:right;font-weight:700;">$6,000 – $12,000</td></tr>
                <tr class="total-row"><td>Total per year</td><td style="text-align:right;">~$8,400 – $17,100</td></tr>
            </table>
            <div style="margin-top:0.9rem;padding:0.85rem 1rem;background:#EAF7F0;border:1.5px solid #A9DFBF;border-radius:12px;font-size:0.9rem;color:#1A5632;line-height:1.6;">
                💡 <strong>Only 15 providers subscribing at $49/month</strong> covers every single running cost. Everything above that is surplus for CCWS.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # BREAK EVEN
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown('<div class="metric-card"><div class="metric-number" style="color:#E74C3C;">15</div><div class="metric-label">Subscribers to cover all running costs</div></div>', unsafe_allow_html=True)
    b2.markdown('<div class="metric-card"><div class="metric-number" style="color:#2E86C1;">$35K</div><div class="metric-label">Revenue/year at 60 subscribers × $49/mo</div></div>', unsafe_allow_html=True)
    b3.markdown('<div class="metric-card"><div class="metric-number" style="color:#27AE60;">2 yrs</div><div class="metric-label">To fully pay back the build cost</div></div>', unsafe_allow_html=True)
    b4.markdown('<div class="metric-card"><div class="metric-number" style="color:#E67E22;">$0</div><div class="metric-label">Cost to seniors — forever free</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # HOW TO GET FUNDED
    st.markdown("### 🏦 How to Pay for the Build — You Don't Need $44K Today")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <div class="section-card" style="border-left:7px solid #E67E22;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.9rem;">🎯 Option 1 — Apply for a Grant</div>
            <div style="color:#2d5a7a;font-size:0.95rem;line-height:2;">
                🔹 <strong>New Horizons for Seniors</strong> — up to $25,000 (Federal)<br>
                🔹 <strong>BC Age-Friendly Grants</strong> — up to $15,000<br>
                🔹 <strong>United Way BC</strong> — digital inclusion fund<br>
                🔹 <strong>Vancouver Foundation</strong> — health equity<br>
                🔹 <strong>Island Health Community Fund</strong> — Victoria
            </div>
            <div style="margin-top:1rem;padding:0.85rem 1rem;background:#FEF5E7;border:1.5px solid #FAD7A0;border-radius:12px;font-size:0.9rem;color:#784212;line-height:1.6;">
                💡 <strong>This demo is your grant application.</strong> Show funders something working — not a PowerPoint.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="section-card" style="border-left:7px solid #2E86C1;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.9rem;">🤝 Option 2 — Partner with an Organization</div>
            <div style="color:#2d5a7a;font-size:0.95rem;line-height:2;">
                🔹 <strong>Island Health</strong> — they need a patient discharge directory<br>
                🔹 <strong>BC Housing</strong> — housing services integration<br>
                🔹 <strong>City of Victoria</strong> — seniors program directory<br>
                🔹 <strong>211 BC</strong> — data partnership<br>
                🔹 <strong>United Way</strong> — co-develop &amp; co-brand
            </div>
            <div style="margin-top:1rem;padding:0.85rem 1rem;background:#EBF5FB;border:1.5px solid #AED6F1;border-radius:12px;font-size:0.9rem;color:#1A4B6E;line-height:1.6;">
                💡 A partner <strong>funds the build</strong>. CCWS keeps ownership and licenses it back to them — and to others.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3-STEP PLAN
    st.markdown("### 🚀 The Plan — 3 Steps")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="highlight-pill">Right Now ✅</div>
            <div class="step-title">Show This Demo</div>
            <div class="step-desc">
                Take this working prototype to grant funders and potential partner organizations.
                A real, clickable app is worth 10× more than a slide deck.<br><br>
                <strong>Goal:</strong> Secure funding or a partnership to build Phase 2.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card" style="border-top-color:#27AE60;">
            <div class="step-num" style="color:#A9DFBF;">02</div>
            <div class="highlight-pill" style="background:linear-gradient(135deg,#27AE60,#1A7A40);">6–12 Months</div>
            <div class="step-title">Launch the Real Platform</div>
            <div class="step-desc">
                Build the production version. Expand to 200+ services.
                Open provider subscriptions with billing.<br><br>
                <strong>Goal:</strong> 15 paying subscribers. Running costs covered. Revenue starts.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card" style="border-top-color:#E67E22;">
            <div class="step-num" style="color:#FAD7A0;">03</div>
            <div class="highlight-pill" style="background:linear-gradient(135deg,#E67E22,#BA4A00);">Year 2+</div>
            <div class="step-title">License to Other Regions</div>
            <div class="step-desc">
                Other cities and health regions pay CCWS to use the same platform under their own brand.
                CCWS earns licensing income without building anything new.<br><br>
                <strong>Goal:</strong> CCWS becomes BC's go-to seniors service platform.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A4B6E,#117A8B);border-radius:20px;padding:2rem 2.4rem;color:white;box-shadow:0 12px 32px rgba(26,75,110,0.3);">
        <div style="font-size:1.4rem;font-weight:800;margin-bottom:1rem;text-align:center;">🏆 About This Project</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
            <div>
                <div style="font-size:0.78rem;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,0.5);margin-bottom:0.5rem;">Built by</div>
                <div style="font-size:1rem;font-weight:800;color:#FFD580;margin-bottom:0.3rem;">APEX Consulting Group</div>
                <div style="font-size:0.92rem;color:rgba(255,255,255,0.8);line-height:1.9;">
                    Qian Li — Project Lead &amp; Service Workflow<br>
                    Nilan Beigi — AI &amp; Data Strategy<br>
                    Sofiya Golagha — Data &amp; Research Analysis<br>
                    Chaz Alec — Strategic Planning &amp; Systems Design
                </div>
            </div>
            <div>
                <div style="font-size:0.78rem;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,0.5);margin-bottom:0.5rem;">Project Details</div>
                <div style="font-size:0.92rem;color:rgba(255,255,255,0.8);line-height:1.9;">
                    <strong style="color:#FFD580;">Sponsor:</strong> Kim Duffus &amp; Joyce Rankin<br>
                    Creating Community Wellness Society<br>
                    <strong style="color:#FFD580;">Faculty Advisor:</strong> Jeff Sawers, BCIT<br>
                    <strong style="color:#FFD580;">Program:</strong> BCIT BITMAN · BSYS-4905 · April 2026
                </div>
            </div>
        </div>
        <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.15);text-align:center;">
            <div style="font-size:0.85rem;color:rgba(255,255,255,0.55);line-height:1.7;">
                This platform was designed and built by APEX Consulting Group as part of the BCIT Business Information Technology Management program.
                CCWS retains full ownership of the platform and all associated intellectual property.
                APEX Consulting Group reserves the right to reference this project in our professional portfolios.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)