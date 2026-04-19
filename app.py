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

    required_cols = ["ServiceName", "Category", "Description", "Location", "Eligibility", "Source"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    if "ServiceID" not in df.columns:
        df["ServiceID"] = [f"SVC-{i+1:04d}" for i in range(len(df))]

    if "FriendlyCategory" not in df.columns:
        df["FriendlyCategory"] = df["Category"]

    if "Subcategory" not in df.columns:
        df["Subcategory"] = ""

    # Location fields
    if "AreaOfVictoria" not in df.columns:
        df["AreaOfVictoria"] = "Victoria"
    if "TransitAccess" not in df.columns:
        df["TransitAccess"] = "Unknown"
    if "AccessMode" not in df.columns:
        df["AccessMode"] = "In person"

    # Contact fields — use CSV values if present
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

    # Physical address
    if "Address" not in df.columns:
        df["Address"] = ""

    # Fee type
    if "FeeType" not in df.columns:
        df["FeeType"] = "Free"

    # Verification
    if "VerificationStatus" not in df.columns:
        df["VerificationStatus"] = "AI reviewed"
    if "LastVerified" not in df.columns:
        df["LastVerified"] = "2026-04-01"
    if "LastUpdatedByProvider" not in df.columns:
        df["LastUpdatedByProvider"] = ""
    if "IsSubscriber" not in df.columns:
        df["IsSubscriber"] = "No"

    # User experience ratings
    for col, default in [("UserRating", 0), ("FriendlyScore", 0), ("EfficientScore", 0),
                         ("EasyToUnderstandScore", 0), ("AccessibilityScore", 0)]:
        if col not in df.columns:
            df[col] = default

    # Fill NaN in string columns
    str_cols = ["Phone", "Email", "Website", "Organization", "ContactPerson",
                "AreaOfVictoria", "TransitAccess", "AccessMode", "FeeType",
                "ChatAvailable", "FAQAvailable", "Address", "VerificationStatus"]
    for col in str_cols:
        df[col] = df[col].fillna("").astype(str)

    return df

base_df = load_data()

# ---------------- LLM Config ----------------
# Deployment-safe: detects Streamlit Cloud / headless environments and disables
# the Ollama localhost call to prevent 60-second hangs. Rule-based search still
# runs as the fallback. To force-enable/disable, set USE_OLLAMA env var.
_deploy_env = (
    os.getenv("STREAMLIT_SERVER_HEADLESS", "").lower() == "true"
    or os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"
    or os.getenv("HOSTNAME", "").startswith("streamlit")
    or os.path.exists("/mount/src")  # Streamlit Cloud container path
)
USE_OLLAMA = os.getenv("USE_OLLAMA", "false" if _deploy_env else "true").lower() == "true"
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
    # Simple user store: {name: {"role": ..., "profile": {...}}}
    "registered_users": {},
    "account_tab_mode": "login",  # "login" or "create"
    "prov_tab_mode": "login",
    "registered_providers": {},
    "faq_topic": None,
    "browse_provider": None,
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
    "computer": ["internet", "email", "smartphone", "tablet", "digital", "technology", "tech help", "online"],
}

USER_FRIENDLY_LEGEND = [
    ("Housing", "Rent help, affordable housing, repairs, assisted living"),
    ("Income Support", "Money help, benefits, subsidies, energy bill assistance"),
    ("Health & Home Care", "Doctors, home care, dental, mental health, caregiver support"),
    ("Emergency", "Crisis lines, legal help, abuse support, alert systems"),
    ("Food & Nutrition", "Meal delivery, food bank, grocery help, community meals"),
    ("Social Activities", "Community groups, friendly visits, dementia support, day programs"),
    ("Taxes & Administrative Support", "Free tax filing, forms, applications, admin help"),
    ("Mobility", "Wheelchairs, walkers, canes, equipment loans"),
    ("Transportation", "Bus pass, HandyDART, volunteer drivers, taxi vouchers"),
    ("Recreation", "Fitness, yoga, aquafit, hobbies, walking clubs"),
    ("Education", "Language classes, life skills, workshops, learning programs"),
    ("Computer Services", "Computer classes, tablet help, internet, email, digital drop-in"),
]

PROFILE_SHORTCUTS = {
    "I need a doctor": "doctor clinic medical home care",
    "I need food help": "food meals delivery grocery nutrition",
    "I need housing help": "housing rent shelter affordable",
    "I need transportation": "transportation bus rides handydart taxi",
    "I need caregiver support": "caregiver support respite friendly visits",
    "I need money help": "income benefits subsidy financial assistance",
}

CATEGORY_KEYWORDS = {
    "Housing": ["housing", "rent", "shelter", "home support", "accessible living", "rental", "affordable housing", "repair", "modification"],
    "Income Support": ["income", "benefits", "financial", "subsidy", "money help", "financial aid", "energy", "utility", "bill"],
    "Health & Home Care": ["doctor", "clinic", "health", "home care", "nurse", "physio", "care", "medical", "dental", "mental health", "respite", "caregiver", "nursing"],
    "Emergency": ["emergency", "urgent", "crisis", "immediate help", "safety", "abuse", "legal", "alert", "exploitation"],
    "Food & Nutrition": ["food", "nutrition", "meals", "meal delivery", "prepared meals", "grocery", "food bank", "hamper", "lunch", "hungry"],
    "Social Activities": ["social", "activities", "community", "friendly visits", "social support", "dementia", "peer support", "visitor", "companionship", "lonely", "isolation"],
    "Taxes & Administrative Support": ["tax", "administrative", "paperwork", "forms", "applications", "filing", "income tax"],
    "Mobility": ["mobility", "walker", "wheelchair", "accessibility", "movement support", "cane", "equipment", "borrow"],
    "Transportation": ["transportation", "bus", "ride", "rides", "transit", "handydart", "taxi", "driver", "transport", "accessible transit", "voucher"],
    "Recreation": ["recreation", "exercise", "wellness", "hobbies", "fitness", "yoga", "aquafit", "sport", "class"],
    "Education": ["education", "learning", "class", "workshop", "training", "literacy", "english", "language", "life skills"],
    "Computer Services": ["computer", "laptop", "tablet", "smartphone", "internet", "email", "digital", "tech help", "online", "technology", "phone help", "ipad", "app"],
}

COST_KEYWORDS = {
    "Free": ["free", "no cost", "fully funded"],
    "Partial pay": ["partial pay", "partial", "subsidized", "partially funded"],
    "Subscription": ["subscription", "monthly fee", "member fee"],
    "Private pay": ["private pay", "paid", "fee for service", "private"],
}

# Category visual identity — icon, colour, Unsplash image
CATEGORY_VISUALS = {
    "Health & Home Care":             ("🏥", "#2778B5", "#EBF5FB", "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400&q=60"),
    "Food & Nutrition":               ("🍽️", "#18A86A", "#E8F8F2", "https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=400&q=60"),
    "Transportation":                 ("🚌", "#E8850A", "#FEF9E7", "https://images.unsplash.com/photo-1570125909232-eb263c188f7e?w=400&q=60"),
    "Housing":                        ("🏠", "#8E44AD", "#F5F0FF", "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=400&q=60"),
    "Social Activities":              ("🤝", "#E74C3C", "#FFF0F5", "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=400&q=60"),
    "Income Support":                 ("💰", "#16A085", "#F0FFF4", "https://images.unsplash.com/photo-1607863680198-23d4b2565df0?w=400&q=60"),
    "Mobility":                       ("♿", "#2980B9", "#F0F4FF", "https://images.unsplash.com/photo-1586773860418-d37222d8fce3?w=400&q=60"),
    "Emergency":                      ("🆘", "#C0392B", "#FFF5F5", "https://images.unsplash.com/photo-1612531386530-97286d97c2d2?w=400&q=60"),
    "Recreation":                     ("🏃", "#27AE60", "#F5FFF0", "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&q=60"),
    "Education":                      ("📚", "#F39C12", "#FFFBF0", "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&q=60"),
    "Computer Services":              ("💻", "#1E8449", "#F0FFF4", "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=400&q=60"),
    "Taxes & Administrative Support": ("📋", "#1A5276", "#F0F8FF", "https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=400&q=60"),
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=8)
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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Fraunces:ital,wght@0,700;1,700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
    font-size: 16px;
}

/* ── PAGE BACKGROUND — warm paper texture ── */
.main {
    background:
        radial-gradient(ellipse at 10% 20%, rgba(255,178,102,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(46,134,193,0.06) 0%, transparent 50%),
        #F6F4EF;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1360px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(190deg, #0F2D45 0%, #1A4B6E 60%, #0D3555 100%);
    border-right: none;
    box-shadow: 4px 0 24px rgba(0,0,0,0.18);
}
[data-testid="stSidebar"] * { color: #E8F4FF !important; }

/* ── INPUTS — rounder, warmer ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #FDFCFA !important;
    color: #1a3a50 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    border: 2px solid #D4C9B8 !important;
    transition: all 0.2s !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: #E8850A !important;
    box-shadow: 0 0 0 3px rgba(232,133,10,0.15) !important;
}

/* ── BUTTONS — organic, warm ── */
.stButton > button {
    background: linear-gradient(135deg, #1A5C8A 0%, #2778B5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.97rem !important;
    padding: 0.65rem 1.1rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 3px 12px rgba(26,92,138,0.28) !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #154E78, #1F65A0) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(26,92,138,0.38) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.85);
    border-radius: 14px;
    padding: 3px;
    gap: 3px;
    border: 1.5px solid #E8E0D5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 11px !important;
    font-weight: 700 !important;
    font-size: 0.93rem !important;
    color: #3d6080 !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1A5C8A, #2778B5) !important;
    color: white !important;
    box-shadow: 0 3px 10px rgba(26,92,138,0.3) !important;
}

/* ── SERVICE CARDS — card with left accent, warm shadow ── */
.service-card {
    background: #FFFFFF;
    border: 1.5px solid #E8E2D8;
    border-left: 5px solid #2778B5;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    color: #1e3a50;
    transition: box-shadow 0.2s, transform 0.2s;
}
.service-card:hover {
    box-shadow: 0 6px 24px rgba(39,120,181,0.14);
    transform: translateY(-1px);
}
.service-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0F2D45;
    margin-bottom: 0.2rem;
    letter-spacing: -0.01em;
}
.service-meta {
    color: #5a7d99;
    font-size: 0.93rem;
    margin-bottom: 0.6rem;
    font-weight: 600;
}

/* ── REC CARDS — top-accent style ── */
.rec-card {
    background: #FFFFFF;
    border: 1.5px solid #D8EAF5;
    border-top: 4px solid #2778B5;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 3px 14px rgba(39,120,181,0.08);
    color: #1e3a50;
    height: 100%;
}
.rec-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0F2D45;
    margin-bottom: 0.2rem;
}
.rec-meta {
    font-size: 0.87rem;
    color: #5a7d99;
    margin-bottom: 0.45rem;
    font-weight: 600;
}
.rec-desc { font-size: 0.93rem; color: #2d4e62; line-height: 1.6; }

/* ── SECTION CARDS ── */
.section-card {
    background: #FFFFFF;
    border: 1.5px solid #E8E2D8;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    color: #1e3a50;
    font-size: 1rem;
    line-height: 1.75;
}
.legend-card {
    background: linear-gradient(135deg, #F0F7FF, #FFFFFF);
    border: 1.5px solid #C8DDF0;
    border-left: 5px solid #2778B5;
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    color: #173951;
    font-size: 0.97rem;
}
.assistant-box {
    background: linear-gradient(135deg, #EBF8F2, #F2FCF6);
    border: 1.5px solid #7EC8A0;
    border-left: 5px solid #18A86A;
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    color: #1a4a33;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
    font-size: 1rem;
    line-height: 1.75;
}
.warning-box {
    background: linear-gradient(135deg, #FFF8EE, #FFFBF5);
    border: 1.5px solid #F0C080;
    border-left: 5px solid #E67E22;
    border-radius: 16px;
    padding: 1rem 1.3rem;
    color: #6E3A0A;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
    font-size: 1rem;
}
.quick-box {
    background: linear-gradient(135deg, #EFF6FF, #F6F9FF);
    border: 1.5px solid #C8DDF0;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    color: #163a56;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

/* ── TAGS & PILLS ── */
.tag {
    display: inline-block;
    background: #EBF5FB;
    color: #1A5276;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
    border: 1px solid #AED6F1;
}

/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #0F2D45 0%, #1A5C8A 100%);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 10px 30px rgba(15,45,69,0.28);
    position: relative;
    overflow: hidden;
}
.hero-title { font-size: 2rem; font-weight: 800; color: #fff; margin-bottom: 0.4rem; line-height: 1.2; }
.hero-sub { font-size: 1rem; color: rgba(255,255,255,0.82); line-height: 1.7; }

/* ── METRIC CARDS ── */
.metric-card {
    background: #FFFFFF;
    border: 1.5px solid #E8E2D8;
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.metric-number { font-size: 2.1rem; font-weight: 800; color: #0F2D45; line-height: 1.1; }
.metric-label { color: #5a7d99; font-size: 0.92rem; margin-top: 0.3rem; line-height: 1.5; }

/* ── STAR RATINGS ── */
.star-row { display:flex; align-items:center; gap:8px; margin-top:0.5rem; flex-wrap:wrap; }
.stars { color: #E8850A; font-size: 1.1rem; letter-spacing: 2px; }
.star-num { font-weight: 800; color: #0F2D45; font-size: 0.9rem; }
.star-label { font-size: 0.82rem; color: #5a7d99; font-weight: 600; }
.rating-grid { display:grid; grid-template-columns:1fr 1fr; gap:5px 14px; margin-top:0.5rem; }
.rating-item { display:flex; align-items:center; gap:8px; font-size:0.82rem; color: #2d5a7a; font-weight:600; }
.rbar-wrap { flex:1; background:#E8E2D8; border-radius:5px; height:6px; overflow:hidden; min-width:36px; }
.rbar { height:6px; border-radius:5px; background: linear-gradient(90deg, #2778B5, #6BB5E8); }

/* ── BUSINESS MODEL ── */
.biz-hero {
    background: linear-gradient(135deg, #0F2D45 0%, #1a647a 100%);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 12px 32px rgba(15,45,69,0.3);
    position: relative;
    overflow: hidden;
}
.biz-hero::after { content:"💼"; position:absolute; right:2rem; top:50%; transform:translateY(-50%); font-size:7rem; opacity:0.09; }
.tier-card { border-radius: 18px; padding: 1.4rem; height: 100%; box-shadow: 0 4px 16px rgba(0,0,0,0.07); }
.tier-free { background:#F8FEFF; border:2px solid #C8DDF0; }
.tier-pro { background:#EBF5FB; border:3px solid #2778B5; }
.tier-enterprise { background:#EFF9F2; border:2px solid #7EC8A0; }
.tier-price { font-size: 2.3rem; font-weight: 800; color: #0F2D45; line-height: 1; margin: 0.5rem 0 1rem; }
.tier-name { font-size: 1.05rem; font-weight: 800; color: #0F2D45; }
.cost-table { width: 100%; border-collapse: collapse; font-size: 0.97rem; color: #1e3a50; }
.cost-table tr { border-bottom: 1.5px solid #E8E2D8; }
.cost-table tr:last-child { border-bottom: none; }
.cost-table td { padding: 0.6rem 0.3rem; line-height: 1.5; }
.cost-table .total-row td { font-weight: 800; font-size: 1.02rem; color: #0F2D45; padding-top: 1rem; border-top: 2.5px solid #2778B5; border-bottom: none; }
.step-card { background:#FFFFFF; border:1.5px solid #E8E2D8; border-top:4px solid #2778B5; border-radius:16px; padding:1.2rem 1.3rem; box-shadow:0 3px 12px rgba(0,0,0,0.05); height:100%; }
.step-num { font-family:'Fraunces',serif; font-size:2.4rem; color:#D4EAF5; line-height:1; margin-bottom:0.3rem; }
.step-title { font-size:1rem; font-weight:800; color:#0F2D45; margin-bottom:0.4rem; }
.step-desc { font-size:0.92rem; color:#2d5a7a; line-height:1.7; }
.highlight-pill { display:inline-block; background:linear-gradient(135deg,#2778B5,#1A5276); color:white; border-radius:999px; padding:0.28rem 0.95rem; font-size:0.8rem; font-weight:700; margin-bottom:0.75rem; letter-spacing:0.3px; }
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
<div style="text-align:center;padding:0.8rem 0 0.6rem;">
    <div style="font-size:1.35rem;font-weight:800;color:#FFD580;letter-spacing:1px;">💙 CCWS Directory</div>
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);margin-top:3px;letter-spacing:0.5px;">Greater Victoria, BC</div>
</div>
""", unsafe_allow_html=True)

# ── Navigation
nav_items = [
    ("🏠", "Home", 0),
    ("🔍", "Find Services", 1),
    ("👤", "My Account", 2),
    ("🏢", "For Providers", 3),
    ("❓", "Help & FAQ", 4),
    ("💼", "Business Model", 5),
]
for icon, label, idx in nav_items:
    if st.sidebar.button(f"{icon}  {label}", key=f"nav_{idx}", use_container_width=True):
        st.session_state.active_tab = idx
        st.rerun()

st.sidebar.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)

# ── Sign in / out — build HTML safely without nested f-string quotes
if st.session_state.user_logged_in or st.session_state.provider_logged_in:
    signin_lines = []
    if st.session_state.user_logged_in:
        signin_lines.append(f"👤 {st.session_state.user_name_demo}")
    if st.session_state.provider_logged_in:
        signin_lines.append(f"🏢 {st.session_state.provider_name_demo}")
    signin_html = "".join(
        f'<div style="font-size:0.88rem;color:#6EE7B7;font-weight:700;margin-top:2px;">{line}</div>'
        for line in signin_lines
    )
    st.sidebar.markdown(
        '<div style="background:rgba(110,231,183,0.08);border:1px solid rgba(110,231,183,0.22);'
        'border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.3rem;">'
        '<div style="font-size:0.7rem;letter-spacing:1.5px;text-transform:uppercase;'
        'color:rgba(255,255,255,0.3);margin-bottom:4px;">Signed in</div>'
        + signin_html +
        '</div>',
        unsafe_allow_html=True
    )
    if st.session_state.user_logged_in:
        if st.sidebar.button("🚪  Sign Out", use_container_width=True, key="sidebar_signout_btn"):
            st.session_state.user_logged_in = False
            st.session_state.user_name_demo = ""
            st.session_state.user_role_demo = "Senior"
            st.session_state.account_tab_mode = "login"
            st.rerun()
else:
    if st.sidebar.button("🔑  Sign In / Create Account", use_container_width=True, key="sidebar_login_btn"):
        st.session_state.active_tab = 2
        st.rerun()

st.sidebar.markdown("---")

# ── Filters — with visible labels so users know what each dropdown does
st.sidebar.markdown(
    '<div style="color:rgba(255,255,255,0.4);font-size:0.7rem;letter-spacing:2px;'
    'text-transform:uppercase;margin-bottom:5px;">🔽 Filter Services</div>',
    unsafe_allow_html=True
)
selected_category = st.sidebar.selectbox("📂 Category", friendly_categories)
selected_location = st.sidebar.selectbox("📍 Location", locations)
selected_cost = st.sidebar.selectbox("💲 Cost", costs)
selected_access_mode = st.sidebar.selectbox("🖥️ Access", access_modes)

engine_query = st.sidebar.text_input(
    "🔍 Quick search",
    value=st.session_state.used_engine_query,
    placeholder="e.g. meals, doctor, bus..."
)
sidebar_search_clicked = False
if st.sidebar.button("🔍  Search", use_container_width=True):
    st.session_state.active_tab = 1
    sidebar_search_clicked = True

# Who are you? — keep functional but hide visually
st.sidebar.markdown('<div style="display:none">', unsafe_allow_html=True)
experience_mode = st.sidebar.radio(
    "Who are you?",
    ["Guest / Public", "Senior / Family", "Provider / Subscriber"],
    index=["Guest / Public", "Senior / Family", "Provider / Subscriber"].index(st.session_state.experience_mode),
    label_visibility="collapsed"
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)
if experience_mode != st.session_state.experience_mode:
    st.session_state.experience_mode = experience_mode
    if experience_mode == "Senior / Family":
        st.session_state.active_tab = 2
    elif experience_mode == "Provider / Subscriber":
        st.session_state.active_tab = 3
    st.rerun()
st.session_state.experience_mode = experience_mode

# ---------------- Header — full-width ----------------
SENIOR_IMG = "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=1400&q=80"
st.markdown(f"""
<div style="position:relative;border-radius:20px;overflow:hidden;margin-bottom:0.9rem;box-shadow:0 10px 32px rgba(15,45,69,0.22);">
    <img src="{SENIOR_IMG}" style="width:100%;height:220px;object-fit:cover;object-position:center 30%;display:block;" alt="Seniors in community"/>
    <div style="position:absolute;inset:0;background:linear-gradient(90deg,rgba(15,45,69,0.92) 0%,rgba(15,45,69,0.60) 55%,rgba(15,45,69,0.15) 100%);"></div>
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;padding:1.8rem 2.2rem;">
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:rgba(255,255,255,0.5);margin-bottom:0.35rem;">Creating Community Wellness Society · Greater Victoria, BC</div>
        <div style="font-size:1.85rem;font-weight:800;color:white;line-height:1.18;margin-bottom:0.35rem;">💙 Find Senior Services — <em style="font-style:italic;color:#A8D8F0;">All in One Place</em></div>
        <div style="font-size:0.93rem;color:rgba(255,255,255,0.8);line-height:1.65;max-width:520px;">One free directory for seniors and caregivers in Greater Victoria. Search in plain words — no special terms, no accounts needed.</div>
        <div style="margin-top:0.75rem;display:flex;gap:0.4rem;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.18rem 0.7rem;font-size:0.78rem;color:white;font-weight:700;">🎙️ Voice Search</span>
            <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.18rem 0.7rem;font-size:0.78rem;color:white;font-weight:700;">🤖 AI Matching</span>
            <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.18rem 0.7rem;font-size:0.78rem;color:white;font-weight:700;">✅ Always Free</span>
            <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.28);border-radius:999px;padding:0.18rem 0.7rem;font-size:0.78rem;color:white;font-weight:700;">📍 40+ Services</span>
            <span style="background:rgba(255,215,128,0.22);border:1px solid rgba(255,215,128,0.5);border-radius:999px;padding:0.18rem 0.7rem;font-size:0.78rem;color:#FFD580;font-weight:700;">🏆 Proof-of-Concept Demo</span>
        </div>
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

# ── FLOATING AI ASSISTANT CHATBOT ──────────────────────────────────────────
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Toggle button — always visible bottom-right via CSS injection
st.markdown("""
<style>
.chat-fab {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 9999;
    width: 62px;
    height: 62px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1A4B6E, #2E86C1);
    box-shadow: 0 6px 22px rgba(26,75,110,0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1.7rem;
    transition: transform 0.2s;
    border: none;
}
.chat-fab:hover { transform: scale(1.1); }
.chat-panel {
    position: fixed;
    bottom: 100px;
    right: 28px;
    z-index: 9998;
    width: 360px;
    max-height: 500px;
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 12px 40px rgba(26,75,110,0.22);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1.5px solid #D6E8F5;
}
</style>
""", unsafe_allow_html=True)

# Chat toggle button
chat_col1, chat_col2 = st.columns([10, 1])
with chat_col2:
    if st.button("💬", key="chat_toggle", help="Chat with AI Assistant"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

# Chat panel — shown inline when open (Streamlit can't do true fixed overlay)
if st.session_state.chat_open:
    st.markdown("""
    <div style="background:#ffffff;border:2px solid #2E86C1;border-radius:20px;margin-bottom:1rem;overflow:hidden;box-shadow:0 8px 32px rgba(26,75,110,0.18);">
        <div style="background:linear-gradient(135deg,#1A4B6E,#2E86C1);padding:1rem 1.2rem;display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:0.6rem;">
                <div style="width:36px;height:36px;background:rgba(255,255,255,0.15);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.2rem;">🤖</div>
                <div>
                    <div style="font-size:1rem;font-weight:800;color:white;">CCWS Assistant</div>
                    <div style="font-size:0.78rem;color:rgba(255,255,255,0.7);">Ask me anything about senior services</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show chat history
    for msg in st.session_state.chat_messages:
        is_user = msg["role"] == "user"
        bg = "#EBF5FB" if is_user else "#E8F8F2"
        border = "#2E86C1" if is_user else "#27AE60"
        align = "flex-end" if is_user else "flex-start"
        icon = "👤" if is_user else "🤖"
        label = "You" if is_user else "Assistant"
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:{align};margin-bottom:0.6rem;">'
            f'<div style="font-size:0.75rem;color:#4a7a99;margin-bottom:2px;">{icon} {label}</div>'
            f'<div style="background:{bg};border:1px solid {border}33;border-radius:14px;'
            f'padding:0.7rem 0.95rem;max-width:85%;font-size:0.95rem;color:#1a3a50;line-height:1.7;">'
            f'{msg["content"]}</div></div>',
            unsafe_allow_html=True
        )

    # Input area
    chat_input_col, chat_send_col = st.columns([5, 1])
    with chat_input_col:
        user_msg = st.text_input(
            "chat_input",
            placeholder="Ask about services, how to find help, transportation...",
            label_visibility="collapsed",
            key="chat_input_box"
        )
    with chat_send_col:
        send_clicked = st.button("➤", key="chat_send", use_container_width=True)

    if send_clicked and user_msg.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_msg.strip()})

        # Build context from services
        service_names = st.session_state.services_df["ServiceName"].tolist()[:20]
        services_context = ", ".join(service_names)

        system_prompt = f"""You are a friendly, helpful assistant for the CCWS Senior Services Directory in Greater Victoria, BC.
Your job is to help seniors, caregivers, and families find the right community services and support.
You speak in plain, simple, warm language — never use jargon.
Keep answers short (2–4 sentences max) unless the person needs detailed steps.
Some services in our directory include: {services_context}.
If someone needs urgent help, always mention: call 2-1-1 (BC211) for free service guidance, or 9-1-1 for emergencies.
Never make up services or phone numbers — if unsure, direct to 2-1-1."""

        try:
            import requests as _req
            import os as _os
            _api_key = _os.environ.get("ANTHROPIC_API_KEY", "")
            _headers = {
                "Content-Type": "application/json",
                "x-api-key": _api_key,
                "anthropic-version": "2023-06-01",
            }
            response = _req.post(
                "https://api.anthropic.com/v1/messages",
                headers=_headers,
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 300,
                    "system": system_prompt,
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages
                    ]
                },
                timeout=15
            )
            if response.status_code == 200:
                reply_text = response.json()["content"][0]["text"]
            else:
                reply_text = f"I'm having trouble connecting right now (error {response.status_code}). Please call **BC211 (dial 2-1-1)** — free, 24 hours, any language."
        except Exception as e:
            reply_text = "I can't connect right now. Please call **BC211 (dial 2-1-1)** — free help finding any service in BC, available 24 hours."

        st.session_state.chat_messages.append({"role": "assistant", "content": reply_text})
        st.rerun()

    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑️  Clear conversation", key="chat_clear"):
            st.session_state.chat_messages = []
            st.rerun()
# ── END CHATBOT ─────────────────────────────────────────────────────────────

# ---------------- Tabs (controlled by active_tab) ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "🔍 Find Services",
    "👤 My Account",
    "🏢 For Providers",
    "❓ Help & FAQ",
    "💼 Business Model",
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


    # ── NUDGE to Find Services tab (no search bar on home)
    _fc = st.columns([3, 1])
    with _fc[0]:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#EBF5FB,#F0F9FF);border:1.5px solid #AED6F1;border-left:5px solid #2E86C1;border-radius:16px;padding:1rem 1.3rem;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">🔍 Ready to search? &nbsp;&nbsp; ❓ Have a question?</div>
            <div style="font-size:0.92rem;color:#2d5a7a;">Click <strong>Find Services</strong> to search. Click <strong>Help &amp; FAQ</strong> for plain-language answers to common questions — including "How do I get meals delivered?" or "What is HandyDART?"</div>
        </div>
        """, unsafe_allow_html=True)
    with _fc[1]:
        if st.button("🔍  Find Services", use_container_width=True, key="home_goto_search"):
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
            Search by voice, type a question in plain language, or browse by category.
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
        ("🚌", "Transportation",     "#FEF9E7", "#E67E22", "Bus, HandyDART, taxi vouchers, volunteer drivers"),
        ("🏠", "Housing",            "#F5F0FF", "#8E44AD", "Rent help, affordable housing, home repairs"),
        ("🤝", "Social Activities",  "#FFF0F5", "#E74C3C", "Community groups, friendly visits, dementia support"),
        ("💰", "Income Support",     "#F0FFF4", "#16A085", "Benefits, financial help, energy assistance"),
        ("♿", "Mobility",           "#F0F4FF", "#2980B9", "Wheelchairs, walkers, equipment loans"),
        ("🆘", "Emergency",         "#FFF5F5", "#C0392B", "Crisis lines, legal help, alert systems"),
        ("🏃", "Recreation",        "#F5FFF0", "#27AE60", "Fitness, yoga, aquafit, community activities"),
        ("📚", "Education",         "#FFFBF0", "#F39C12", "Language classes, life skills, workshops"),
        ("💻", "Computer Services", "#F0FFF4", "#1E8449", "Computer classes, tablet help, internet, digital drop-in"),
        ("📋", "Taxes & Administrative Support", "#F0F8FF", "#1A5276", "Free tax filing, forms, administrative help"),
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

    # ── SECTION HEADER
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A4B6E 0%,#2E86C1 100%);border-radius:20px;padding:1.4rem 2rem;margin-bottom:1.2rem;box-shadow:0 8px 24px rgba(26,75,110,0.2);">
        <div style="font-size:1.6rem;font-weight:800;color:white;margin-bottom:0.2rem;">🔍 Find Services</div>
        <div style="font-size:0.95rem;color:rgba(255,255,255,0.78);">Search in plain words — type, tap a button, or speak. No special terms needed.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── ONE UNIFIED SEARCH CARD ──────────────────────────────────────────────
    st.markdown("""
    <div style="background:#ffffff;border:2px solid #D6E8F5;border-radius:20px;padding:1.5rem 1.6rem;margin-bottom:1rem;box-shadow:0 4px 18px rgba(0,0,0,0.06);">
        <div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:0.5rem;">
            How would you like to search?
        </div>
        <div style="font-size:0.92rem;color:#4a7a99;margin-bottom:0.9rem;">
            Type what you need below, <strong>or</strong> press the microphone to speak, <strong>or</strong> tap one of the common needs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: text input + Search button + mic button — all in one row
    col_input, col_search, col_mic = st.columns([5, 1, 1])
    with col_input:
        typed_query = st.text_input(
            "Search",
            value=st.session_state.search_query,
            placeholder='e.g. "I need meals delivered"  or  "help with transportation"  or  "doctor near me"',
            label_visibility="collapsed",
        )
    with col_search:
        typed_search_clicked = st.button("🔍  Search", use_container_width=True, key="main_typed_search")
    with col_mic:
        reset_clicked = st.button("↺  Clear", use_container_width=True, key="main_clear")

    # Voice input — sits neatly below the input row, compact
    voice_row1, voice_row2 = st.columns([3, 1])
    with voice_row1:
        if VOICE_AVAILABLE:
            voice_text_input = speech_to_text(
                language="en",
                start_prompt="🎤  Tap to Speak",
                stop_prompt="⏹  Stop",
                just_once=True,
                use_container_width=True,
                key="main_voice_search",
            )
            if voice_text_input:
                st.session_state.voice_text_display = voice_text_input
        else:
            st.caption("🎤 Voice search not available in this environment.")
    with voice_row2:
        voice_search_clicked = st.button("🎤  Search by Voice", use_container_width=True, key="voice_go")

    # Show what was captured by voice (compact, inline)
    if st.session_state.voice_text_display:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#EBF5FB,#F0F9FF);border:1.5px dashed #7ABFE8;border-radius:12px;padding:0.6rem 1rem;margin-top:0.4rem;font-size:0.95rem;color:#1A4B6E;">'
            '🎤 <strong>You said:</strong> &nbsp;' + str(st.session_state.voice_text_display) +
            '</div>',
            unsafe_allow_html=True
        )

    # Divider
    st.markdown('<div style="border-top:1.5px solid #EEF5FB;margin:0.9rem 0 0.7rem;"></div>', unsafe_allow_html=True)

    # Quick-tap shortcuts — 3 columns, compact
    st.markdown('<div style="font-size:0.88rem;font-weight:700;color:#4a7a99;margin-bottom:0.5rem;">Or tap a common need:</div>', unsafe_allow_html=True)
    shortcut_cols = st.columns(3)
    shortcut_items = list(PROFILE_SHORTCUTS.items())
    for i, (label, query_value) in enumerate(shortcut_items):
        with shortcut_cols[i % 3]:
            if st.button(label, key=f"shortcut_{i}", use_container_width=True):
                execute_search(query_value, "")

    # Active filters pill row
    active_filters = []
    if selected_category != "All": active_filters.append(f"📂 {selected_category}")
    if selected_location != "All": active_filters.append(f"📍 {selected_location}")
    if selected_cost != "All": active_filters.append(f"💲 {selected_cost}")
    if selected_access_mode != "All": active_filters.append(f"🖥️ {selected_access_mode}")
    if active_filters:
        filter_pills = " &nbsp; ".join([f'<span style="background:#EBF5FB;color:#1A5276;border:1px solid #AED6F1;border-radius:999px;padding:0.2rem 0.75rem;font-size:0.82rem;font-weight:700;">{f}</span>' for f in active_filters])
        st.markdown(f"""
        <div style="background:#EBF5FB;border:1.5px solid #AED6F1;border-radius:12px;padding:0.6rem 1rem;margin-top:0.6rem;display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">
            <span style="font-size:0.82rem;font-weight:800;color:#1A5276;">Active filters:</span>
            {filter_pills}
            <span style="font-size:0.78rem;color:#4a7a99;">(Change in sidebar ←)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:0.5rem;"></div>', unsafe_allow_html=True)
    # ── END UNIFIED SEARCH CARD ──────────────────────────────────────────────

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

    # ── UNIFIED AI RESULT BOX (search summary + AI reply + top picks in one card)
    if st.session_state.searched and (st.session_state.final_query or st.session_state.assistant_reply or not st.session_state.recommended.empty):
        _query_line = f'<div style="font-size:0.85rem;color:#4a7a99;margin-bottom:0.2rem;">You searched for:</div><div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">"{st.session_state.final_query}"</div>' if st.session_state.final_query else ""
        _summary_line = f'<div style="font-size:0.88rem;color:#27AE60;margin-bottom:0.6rem;">🎯 {st.session_state.assistant_summary}</div>' if st.session_state.assistant_summary else ""
        _reply_line = f'<div style="display:flex;align-items:flex-start;gap:0.6rem;padding:0.7rem 0.9rem;background:rgba(255,255,255,0.6);border-radius:12px;margin-bottom:0.4rem;"><span style="font-size:1.2rem;flex-shrink:0;">🤖</span><div style="font-size:0.95rem;color:#1a4a33;line-height:1.7;">{st.session_state.assistant_reply}</div></div>' if st.session_state.assistant_reply else ""
        st.markdown(
            '<div style="background:linear-gradient(135deg,#E8F8F2,#F0FBF5);border:1.5px solid #82C9A5;border-left:5px solid #1DB87A;border-radius:18px;padding:1.1rem 1.4rem;margin:0.8rem 0;">' +
            _query_line + _summary_line + _reply_line +
            '</div>',
            unsafe_allow_html=True
        )

    if not st.session_state.recommended.empty:
        st.markdown('<div style="font-size:1.05rem;font-weight:800;color:#1A4B6E;margin:0.2rem 0 0.7rem;">⭐ Top Picks for You</div>', unsafe_allow_html=True)
        rec_cols = st.columns(min(3, len(st.session_state.recommended)))
        for i, (_, row) in enumerate(st.session_state.recommended.iterrows()):
            with rec_cols[i]:
                _rating = rating_block_html(row)
                _verif = verification_label(row)
                _subcat = row['Subcategory'] if str(row['Subcategory']).strip() else 'General service'
                _fee = row.get('FeeType', row.get('CostType', ''))
                _contact_icons = ""
                if str(row.get("Phone","")).strip(): _contact_icons += "📞 "
                if str(row.get("Email","")).strip(): _contact_icons += "✉️ "
                if str(row.get("Website","")).strip(): _contact_icons += "🌐 "
                if str(row.get("ChatAvailable","")) == "Yes": _contact_icons += "💬 "
                st.markdown(
                    '<div class="rec-card">' +
                    f'<div class="rec-title">{row["ServiceName"]}</div>' +
                    f'<div class="rec-meta">{row["FriendlyCategory"]} &nbsp;·&nbsp; {row["Location"]} &nbsp;·&nbsp; <strong style="color:#27AE60">{_fee}</strong></div>' +
                    f'<div class="rec-desc">{row["Description"]}</div>' +
                    _rating +
                    (f'<div style="margin-top:0.5rem;font-size:0.85rem;color:#4a7a99;">Contact: {_contact_icons}</div>' if _contact_icons else '') +
                    f'<div style="margin-top:0.5rem;"><span class="tag">{_subcat}</span></div>' +
                    '</div>',
                    unsafe_allow_html=True
                )

    results = st.session_state.results
    if not results.empty:
        st.markdown(f"""
        <div style="background:#ffffff;border:1.5px solid #D6E8F5;border-radius:14px;padding:0.8rem 1.2rem;margin:1rem 0 0.6rem;display:flex;align-items:center;justify-content:space-between;">
            <div style="font-size:1rem;font-weight:800;color:#1A4B6E;">📋 All Matching Services</div>
            <div style="background:#2E86C1;color:white;border-radius:999px;padding:0.2rem 0.8rem;font-size:0.9rem;font-weight:700;">{len(results)} found</div>
        </div>
        """, unsafe_allow_html=True)

        for _, row in results.iterrows():
            _rating = rating_block_html(row)
            _verif = verification_label(row)
            _subcat = row['Subcategory'] if str(row['Subcategory']).strip() else 'General'

            # Build contact info row
            contact_parts = []
            if str(row.get("Phone","")).strip() and str(row.get("Phone","")) not in ["nan", ""]:
                contact_parts.append(f'📞 <strong>{row["Phone"]}</strong>')
            if str(row.get("Email","")).strip() and str(row.get("Email","")) not in ["nan", ""]:
                contact_parts.append(f'✉️ {row["Email"]}')
            if str(row.get("ChatAvailable","")) == "Yes":
                contact_parts.append("💬 Chat available")
            if str(row.get("FAQAvailable","")) == "Yes":
                contact_parts.append("❓ FAQ available")
            contact_html = " &nbsp;·&nbsp; ".join(contact_parts) if contact_parts else "Contact info not listed"

            # Website link
            _website = str(row.get("Website","")).strip()
            website_html = f'<a href="{_website}" target="_blank" style="color:#2E86C1;font-weight:600;text-decoration:none;">🌐 Visit website</a>' if _website and _website not in ["nan", ""] else ""

            # Address + Google Maps link
            _address = str(row.get("Address","")).strip()
            maps_html = ""
            if _address and _address not in ["nan", ""] and "Online" not in _address and "Phone" not in _address:
                maps_query = _address.replace(" ", "+")
                maps_html = f'<a href="https://www.google.com/maps/search/?api=1&query={maps_query}" target="_blank" style="color:#E67E22;font-weight:600;text-decoration:none;font-size:0.88rem;">🗺️ View on Google Maps</a>'

            # Category visual banner
            cat_key = str(row.get("FriendlyCategory", row.get("Category", ""))).strip()
            cat_vis = CATEGORY_VISUALS.get(cat_key, ("🔵", "#2778B5", "#EBF5FB", ""))
            cat_icon, cat_color, cat_bg, cat_img = cat_vis

            # Fee type badge color
            fee = row.get("FeeType", "")
            fee_color = "#18A86A" if fee == "Free" else ("#E67E22" if fee == "Partial pay" else "#8E44AD" if fee == "Subscription" else "#E74C3C")

            # Image banner
            img_banner = ""
            if cat_img:
                img_banner = (
                    f'<div style="position:relative;border-radius:10px;overflow:hidden;height:72px;margin-bottom:0.7rem;">'
                    f'<img src="{cat_img}" style="width:100%;height:72px;object-fit:cover;display:block;" loading="lazy" alt="{cat_key}"/>'
                    f'<div style="position:absolute;inset:0;background:linear-gradient(90deg,rgba(15,45,69,0.82) 0%,rgba(15,45,69,0.4) 55%,transparent 100%);"></div>'
                    f'<div style="position:absolute;top:50%;left:0.8rem;transform:translateY(-50%);display:flex;align-items:center;gap:0.45rem;">'
                    f'<span style="font-size:1.2rem;">{cat_icon}</span>'
                    f'<span style="font-size:0.8rem;font-weight:700;color:white;">{cat_key}</span>'
                    f'</div>'
                    f'<div style="position:absolute;top:50%;right:0.8rem;transform:translateY(-50%);">'
                    f'<span style="background:{fee_color};color:white;border-radius:999px;padding:0.15rem 0.7rem;font-size:0.78rem;font-weight:800;">{fee}</span>'
                    f'</div>'
                    f'</div>'
                )

            st.markdown(
                '<div class="service-card" style="border-left-color:' + fee_color + ';padding-top:0.85rem;">'
                + img_banner +
                f'<div style="font-size:1.18rem;font-weight:800;color:#0F2D45;margin-bottom:0.2rem;">{row["ServiceName"]}</div>'
                f'<div class="service-meta">'
                f'<span>📍 {row["AreaOfVictoria"]}</span>'
                f' &nbsp;·&nbsp; <span>🖥️ {row["AccessMode"]}</span>'
                f' &nbsp;·&nbsp; <span>🚌 Transit: {row["TransitAccess"]}</span>'
                f'</div>'
                f'<div style="margin-bottom:0.6rem;font-size:1rem;color:#2d4e62;line-height:1.75;">{row["Description"]}</div>'
                '<div style="border-top:1px solid #E8E2D8;margin:0.5rem 0 0.4rem;"></div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem 1rem;font-size:0.87rem;color:#2d5a7a;margin-bottom:0.4rem;">'
                f'<div><strong>🏢</strong> {row["Organization"] or row["Provider"] or "N/A"}</div>'
                f'<div><strong>👤</strong> {row["ContactPerson"] or "Not listed"}</div>'
                f'<div style="grid-column:span 2;">{contact_html}</div>'
                + (f'<div style="grid-column:span 2;"><strong>📌</strong> {_address} &nbsp; {maps_html}</div>' if _address and _address not in ["nan",""] else '') +
                (f'<div>{website_html}</div>' if website_html else '') +
                f'</div>'
                + _rating +
                f'<div style="margin-top:0.4rem;font-size:0.79rem;color:#5a7d99;">{_verif}</div>'
                '</div>',
                unsafe_allow_html=True
            )

            action_col1, action_col2 = st.columns([1, 1])
            with action_col1:
                if st.session_state.user_logged_in:
                    if st.button(f"💾 Save This Service", key=f"save_{row['ServiceID']}", use_container_width=True):
                        save_service(row["ServiceID"])
                        st.success(f"✅ {row['ServiceName']} saved to My Account!")
                else:
                    st.caption("👤 Log in to My Account to save services")
            with action_col2:
                if st.session_state.user_logged_in:
                    with st.expander(f"📞 Contact {row['ServiceName']}"):
                        request_message = st.text_area("Your message", placeholder="Hello, I would like more information.", key=f"msg_{row['ServiceID']}")
                        request_contact = st.selectbox("How should they contact you?", ["Phone", "Email"], key=f"contact_pref_{row['ServiceID']}")
                        if st.button("📨 Send Message", key=f"send_req_{row['ServiceID']}", use_container_width=True):
                            add_user_request(row, request_message.strip() or "Requesting more information.", request_contact)
                            st.success("✅ Message sent! The provider will contact you soon.")
                else:
                    st.caption("👤 Log in to My Account to contact providers")

    elif st.session_state.searched:
        st.markdown('<div class="warning-box" style="text-align:center;margin-top:1.5rem;"><div style="font-size:1.1rem;font-weight:800;margin-bottom:0.5rem;">No services found for that search 😔</div><div style="font-size:0.97rem;line-height:1.7;">Try simpler words like: <strong>doctor, meals, bus, housing, food, help at home, computer help</strong></div></div>', unsafe_allow_html=True)

    # ── BROWSE BY PROVIDER — always shown at bottom of search results
    if not results.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#FFFFFF;border:1.5px solid #E8E2D8;border-left:5px solid #E8850A;border-radius:14px;padding:0.9rem 1.2rem;margin-bottom:0.8rem;">
            <div style="font-size:1rem;font-weight:800;color:#0F2D45;">🏢 See All Services from One Provider</div>
            <div style="font-size:0.88rem;color:#5a7d99;margin-top:2px;">Some organizations offer more than one service. Pick a provider to see everything they offer.</div>
        </div>
        """, unsafe_allow_html=True)

        # Get unique providers from current results
        result_orgs = sorted(results["Organization"].dropna().unique().tolist())
        result_orgs = [o for o in result_orgs if str(o).strip() and str(o) != "nan"]

        if result_orgs:
            org_cols = st.columns(min(4, len(result_orgs)))
            for i, org in enumerate(result_orgs[:8]):
                org_count = len(st.session_state.services_df[
                    st.session_state.services_df["Organization"].str.lower() == org.lower()
                ])
                with org_cols[i % min(4, len(result_orgs))]:
                    if st.button(f"🏥 {org[:22]}\n({org_count} service{'s' if org_count != 1 else ''})", key=f"org_browse_{i}", use_container_width=True):
                        st.session_state["browse_provider"] = org
                        st.rerun()

            # Show selected provider's all services
            if st.session_state.get("browse_provider"):
                bp = st.session_state["browse_provider"]
                bp_services = st.session_state.services_df[
                    st.session_state.services_df["Organization"].str.lower() == bp.lower()
                ].copy()
                st.markdown(f"""
                <div style="background:#EFF6FF;border:1.5px solid #C8DDF0;border-radius:12px;padding:0.7rem 1.1rem;margin:0.8rem 0 0.5rem;display:flex;align-items:center;justify-content:space-between;">
                    <div style="font-weight:800;color:#0F2D45;">All services from: {bp}</div>
                    <div style="background:#2778B5;color:white;border-radius:999px;padding:0.15rem 0.7rem;font-size:0.85rem;font-weight:700;">{len(bp_services)} total</div>
                </div>
                """, unsafe_allow_html=True)
                for _, row in bp_services.iterrows():
                    fee = row.get("FeeType","")
                    fee_color = "#18A86A" if fee=="Free" else "#E67E22" if fee=="Partial pay" else "#8E44AD" if fee=="Subscription" else "#E74C3C"
                    _phone = str(row.get("Phone","")).strip()
                    _addr = str(row.get("Address","")).strip()
                    maps_link = ""
                    if _addr and _addr not in ["nan",""] and "Online" not in _addr:
                        maps_link = f'&nbsp;·&nbsp;<a href="https://www.google.com/maps/search/?api=1&query={_addr.replace(" ","+")}" target="_blank" style="color:#E67E22;text-decoration:none;font-weight:600;">🗺️ Map</a>'
                    st.markdown(
                        f'<div class="service-card" style="border-left-color:{fee_color};margin-bottom:0.6rem;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">'
                        f'<div class="service-title" style="font-size:1rem;">{row["ServiceName"]}</div>'
                        f'<span style="background:{fee_color}18;color:{fee_color};border:1.5px solid {fee_color}44;border-radius:999px;padding:0.1rem 0.65rem;font-size:0.82rem;font-weight:800;">{fee}</span>'
                        f'</div>'
                        f'<div class="service-meta">{row["FriendlyCategory"]} &nbsp;·&nbsp; {row.get("AreaOfVictoria","Victoria")}</div>'
                        f'<div style="font-size:0.92rem;color:#2d4e62;line-height:1.6;margin-bottom:0.3rem;">{row["Description"]}</div>'
                        f'<div style="font-size:0.84rem;color:#5a7d99;">{"📞 " + _phone if _phone and _phone not in ["nan",""] else ""}{maps_link}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                if st.button("✕  Close Provider View", key="close_provider_view"):
                    st.session_state["browse_provider"] = None
                    st.rerun()

    if not st.session_state.searched:
        # Clean "not searched yet" state — just show Browse by Provider from all services
        st.markdown("""
        <div style="background:#FFFFFF;border:1.5px solid #E8E2D8;border-left:5px solid #2778B5;border-radius:14px;padding:0.9rem 1.2rem;margin-bottom:0.8rem;">
            <div style="font-size:1rem;font-weight:800;color:#0F2D45;">🏢 Browse Services by Provider</div>
            <div style="font-size:0.88rem;color:#5a7d99;margin-top:2px;">See all services from a specific organization — some providers offer more than one.</div>
        </div>
        """, unsafe_allow_html=True)

        all_orgs = sorted(st.session_state.services_df["Organization"].dropna().unique().tolist())
        all_orgs = [o for o in all_orgs if str(o).strip() and str(o) != "nan"]
        selected_org = st.selectbox("Choose a provider organization:", ["— Select a provider —"] + all_orgs, label_visibility="collapsed")

        if selected_org and selected_org != "— Select a provider —":
            org_services = st.session_state.services_df[
                st.session_state.services_df["Organization"].str.lower() == selected_org.lower()
            ].copy()
            st.markdown(f"""
            <div style="background:#EFF6FF;border:1.5px solid #C8DDF0;border-radius:12px;padding:0.7rem 1rem;margin-bottom:0.6rem;display:flex;align-items:center;justify-content:space-between;">
                <div style="font-weight:800;color:#0F2D45;font-size:0.97rem;">🏥 {selected_org}</div>
                <div style="background:#2778B5;color:white;border-radius:999px;padding:0.15rem 0.7rem;font-size:0.85rem;font-weight:700;">{len(org_services)} service{"s" if len(org_services)!=1 else ""}</div>
            </div>
            """, unsafe_allow_html=True)
            for _, row in org_services.iterrows():
                fee = row.get("FeeType","")
                fee_color = "#18A86A" if fee=="Free" else "#E67E22" if fee=="Partial pay" else "#8E44AD" if fee=="Subscription" else "#E74C3C"
                _phone = str(row.get("Phone","")).strip()
                _addr = str(row.get("Address","")).strip()
                maps_link = ""
                if _addr and _addr not in ["nan",""] and "Online" not in _addr:
                    maps_link = f'&nbsp;·&nbsp;<a href="https://www.google.com/maps/search/?api=1&query={_addr.replace(" ","+")}" target="_blank" style="color:#E67E22;font-weight:600;text-decoration:none;">🗺️ Map</a>'
                st.markdown(
                    f'<div class="service-card" style="border-left-color:{fee_color};margin-bottom:0.7rem;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.3rem;">'
                    f'<div class="service-title" style="font-size:1.05rem;">{row["ServiceName"]}</div>'
                    f'<span style="background:{fee_color}18;color:{fee_color};border:1.5px solid {fee_color}44;border-radius:999px;padding:0.12rem 0.7rem;font-size:0.82rem;font-weight:800;">{fee}</span>'
                    f'</div>'
                    f'<div class="service-meta">{row["FriendlyCategory"]} &nbsp;·&nbsp; {row.get("AreaOfVictoria","Victoria")}</div>'
                    f'<div style="font-size:0.93rem;color:#2d4e62;line-height:1.65;margin-bottom:0.3rem;">{row["Description"]}</div>'
                    f'<div style="font-size:0.85rem;color:#5a7d99;">{"📞 " + _phone if _phone and _phone not in ["nan",""] else ""}{maps_link}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ---------------- MY ACCOUNT (Senior Portal) ----------------
with tab3:

    if not st.session_state.user_logged_in:

        # ── PAGE HEADER
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 1.2rem;">
            <div style="font-size:2rem;font-weight:800;color:#1A4B6E;margin-bottom:0.4rem;">👤 My Account</div>
            <div style="font-size:1rem;color:#4a7a99;max-width:600px;margin:0 auto;line-height:1.8;">
                Save services, build a shortlist, and message providers — all in one place. No password needed.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── MODE TOGGLE: Sign In / Create Account
        toggle_col1, toggle_col2 = st.columns(2)
        with toggle_col1:
            if st.button(
                "🔑  Sign In" + (" ◀" if st.session_state.account_tab_mode == "login" else ""),
                use_container_width=True,
                key="toggle_login",
                type="primary" if st.session_state.account_tab_mode == "login" else "secondary",
            ):
                st.session_state.account_tab_mode = "login"
                st.rerun()
        with toggle_col2:
            if st.button(
                "✨  Create Account" + (" ◀" if st.session_state.account_tab_mode == "create" else ""),
                use_container_width=True,
                key="toggle_create",
                type="primary" if st.session_state.account_tab_mode == "create" else "secondary",
            ):
                st.session_state.account_tab_mode = "create"
                st.rerun()

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:

            # ════════════════════════════════
            #  SIGN IN panel
            # ════════════════════════════════
            if st.session_state.account_tab_mode == "login":
                has_users = len(st.session_state.registered_users) > 0
                st.markdown(f"""
                <div style="background:#ffffff;border:2px solid #2E86C1;border-radius:20px;padding:2rem 1.8rem;box-shadow:0 6px 20px rgba(46,134,193,0.12);text-align:center;">
                    <div style="font-size:2.2rem;margin-bottom:0.6rem;">🔑</div>
                    <div style="font-size:1.15rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">Welcome back!</div>
                    <div style="font-size:0.9rem;color:#4a7a99;margin-bottom:1.2rem;">{"Select your name below to sign in." if has_users else "No accounts yet — create one first."}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                if has_users:
                    registered_names = list(st.session_state.registered_users.keys())
                    login_name = st.selectbox(
                        "Select your name",
                        registered_names,
                        key="login_name_select"
                    )
                    st.markdown(f"""
                    <div style="background:#EBF5FB;border:1px solid #AED6F1;border-radius:10px;padding:0.7rem 1rem;font-size:0.9rem;color:#2d5a7a;margin-bottom:0.8rem;">
                        Role: <strong>{st.session_state.registered_users.get(login_name, {}).get("role", "")}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🔑  Sign In", use_container_width=True, key="do_login"):
                        user_data = st.session_state.registered_users[login_name]
                        st.session_state.user_logged_in = True
                        st.session_state.user_name_demo = login_name
                        st.session_state.user_role_demo = user_data["role"]
                        st.session_state.user_profile = user_data["profile"].copy()
                        st.session_state.user_profile["full_name"] = login_name
                        st.success(f"✅ Welcome back, {login_name}!")
                        st.rerun()
                else:
                    st.info("No accounts found. Use **Create Account** to register first.")
                    if st.button("✨  Go to Create Account", use_container_width=True, key="go_create"):
                        st.session_state.account_tab_mode = "create"
                        st.rerun()

            # ════════════════════════════════
            #  CREATE ACCOUNT panel
            # ════════════════════════════════
            else:
                st.markdown("""
                <div style="background:#ffffff;border:2px solid #27AE60;border-radius:20px;padding:2rem 1.8rem;box-shadow:0 6px 20px rgba(39,174,96,0.1);text-align:center;">
                    <div style="font-size:2.2rem;margin-bottom:0.6rem;">👋</div>
                    <div style="font-size:1.15rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">Create your free account</div>
                    <div style="font-size:0.9rem;color:#4a7a99;margin-bottom:0.2rem;">No password needed. No personal health information collected.</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                new_name = st.text_input("Your name", placeholder="e.g. Margaret or John's Family", key="create_name_input")
                new_role = st.selectbox("I am a:", ["Senior", "Family Member", "Caregiver"], key="create_role_select")

                # Check if name already taken
                name_taken = new_name.strip() in st.session_state.registered_users if new_name.strip() else False
                if name_taken:
                    st.warning(f"⚠️ The name **{new_name.strip()}** already has an account. Use Sign In instead.")

                if st.button("✅  Create My Account & Sign In", use_container_width=True, key="do_create"):
                    if not new_name.strip():
                        st.warning("Please enter your name to continue.")
                    elif name_taken:
                        st.error("That name is already registered. Please sign in or use a different name.")
                    else:
                        # Register new user
                        new_profile = {
                            "full_name": new_name.strip(),
                            "role": new_role,
                            "location": "Victoria",
                            "preferred_contact": "Phone",
                            "cost_preference": "Public",
                            "mobility_needs": "",
                            "transportation_needs": "",
                            "support_interests": [],
                            "notes": "",
                        }
                        st.session_state.registered_users[new_name.strip()] = {
                            "role": new_role,
                            "profile": new_profile.copy(),
                        }
                        st.session_state.user_logged_in = True
                        st.session_state.user_name_demo = new_name.strip()
                        st.session_state.user_role_demo = new_role
                        st.session_state.user_profile = new_profile
                        st.success(f"✅ Account created! Welcome, {new_name.strip()}!")
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        feat1, feat2, feat3 = st.columns(3)
        with feat1:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">💾</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Save Services</div><div style="font-size:0.9rem;color:#4a7a99;">Keep a shortlist of services you want to remember or come back to.</div></div>""", unsafe_allow_html=True)
        with feat2:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">📨</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Message Providers</div><div style="font-size:0.9rem;color:#4a7a99;">Send a message directly to a service provider and they'll get back to you.</div></div>""", unsafe_allow_html=True)
        with feat3:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">👤</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Your Profile</div><div style="font-size:0.9rem;color:#4a7a99;">Set your preferences so we can find the most relevant services for you.</div></div>""", unsafe_allow_html=True)

    else:
        # ── LOGGED IN: header + sign out
        signout_col1, signout_col2 = st.columns([3, 1])
        with signout_col1:
            st.markdown(f"""
            <div class="assistant-box" style="display:flex;align-items:center;gap:1rem;">
                <div style="font-size:2.5rem;">👋</div>
                <div>
                    <div style="font-size:1.2rem;font-weight:800;color:#1A4B6E;">Welcome back, {st.session_state.user_profile["full_name"]}!</div>
                    <div style="font-size:0.95rem;color:#4a7a99;">Signed in as: <strong>{st.session_state.user_profile["role"]}</strong> &nbsp;·&nbsp; Greater Victoria</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with signout_col2:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("🚪  Sign Out", use_container_width=True, key="signout_btn"):
                st.session_state.user_logged_in = False
                st.session_state.user_name_demo = ""
                st.session_state.user_role_demo = "Senior"
                st.session_state.account_tab_mode = "login"
                st.rerun()

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
                    location = st.text_input(
                        "My area of Victoria",
                        value=st.session_state.user_profile["location"],
                        placeholder="e.g. Saanich, Oak Bay, Victoria Downtown"
                    )
                    home_address = st.text_input(
                        "📌 My home address (optional — used to find nearby services)",
                        value=st.session_state.user_profile.get("home_address", ""),
                        placeholder="e.g. 123 Oak Bay Ave, Victoria BC"
                    )
                    preferred_contact = st.selectbox(
                        "Best way to reach me:",
                        ["Phone", "Email"],
                        index=["Phone", "Email"].index(st.session_state.user_profile["preferred_contact"])
                    )
                with up2:
                    cost_preference = st.selectbox(
                        "Cost preference:",
                        ["Public / Free", "Partial pay", "Private pay", "Any"],
                        index=["Public / Free", "Partial pay", "Private pay", "Any"].index(
                            st.session_state.user_profile.get("cost_preference_new",
                            "Public / Free" if st.session_state.user_profile["cost_preference"] == "Public" else "Any")
                        ) if st.session_state.user_profile.get("cost_preference_new") else 0
                    )
                    mobility_needs = st.text_input("Mobility or accessibility needs (optional)", value=st.session_state.user_profile["mobility_needs"])
                    transportation_needs = st.text_input("Transportation needs (optional)", value=st.session_state.user_profile["transportation_needs"])
                    support_interests = st.multiselect(
                        "What kinds of support are you looking for?",
                        ["Doctor / Clinic", "Food & Nutrition", "Housing", "Transportation",
                         "Community Support", "Financial Help", "Safety & Protection",
                         "Computer Help", "Recreation", "Education"],
                        default=st.session_state.user_profile["support_interests"]
                    )
                notes = st.text_area(
                    "Any other notes (optional)",
                    value=st.session_state.user_profile["notes"],
                    placeholder="General preferences only — no medical records collected."
                )
                profile_save = st.form_submit_button("💾  Save My Profile", use_container_width=True)

                if profile_save:
                    new_profile_data = {
                        "full_name": full_name,
                        "role": role,
                        "location": location,
                        "home_address": home_address,
                        "preferred_contact": preferred_contact,
                        "cost_preference": "Public",
                        "cost_preference_new": cost_preference,
                        "mobility_needs": mobility_needs,
                        "transportation_needs": transportation_needs,
                        "support_interests": support_interests,
                        "notes": notes,
                    }
                    st.session_state.user_profile = new_profile_data
                    # Persist back so next login restores the profile
                    current_name = st.session_state.user_name_demo
                    if current_name in st.session_state.registered_users:
                        st.session_state.registered_users[current_name]["role"] = role
                        st.session_state.registered_users[current_name]["profile"] = new_profile_data.copy()
                    st.success("✅ Profile saved.")

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

    if not st.session_state.provider_logged_in:

        # Header
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 1.2rem;">
            <div style="font-size:2rem;font-weight:800;color:#1A4B6E;margin-bottom:0.4rem;">🏢 For Providers</div>
            <div style="font-size:1rem;color:#4a7a99;max-width:600px;margin:0 auto;line-height:1.8;">
                List your services where seniors are already looking. Manage your listings and respond to contact requests — all in one place.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Login toggle
        prov_toggle1, prov_toggle2 = st.columns(2)
        with prov_toggle1:
            if st.button(
                "🔑  Sign In" + (" ◀" if st.session_state.get("prov_tab_mode","login") == "login" else ""),
                use_container_width=True, key="ptog_login",
                type="primary" if st.session_state.get("prov_tab_mode","login") == "login" else "secondary"
            ):
                st.session_state.prov_tab_mode = "login"
                st.rerun()
        with prov_toggle2:
            if st.button(
                "✨  Register as Provider" + (" ◀" if st.session_state.get("prov_tab_mode","login") == "register" else ""),
                use_container_width=True, key="ptog_reg",
                type="primary" if st.session_state.get("prov_tab_mode","login") == "register" else "secondary"
            ):
                st.session_state.prov_tab_mode = "register"
                st.rerun()

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

        pcol_a, pcol_b, pcol_c = st.columns([1, 2, 1])
        with pcol_b:
            prov_mode = st.session_state.get("prov_tab_mode", "login")

            if prov_mode == "login":
                known_providers = list(st.session_state.get("registered_providers", {}).keys())
                st.markdown("""
                <div style="background:#ffffff;border:2px solid #2E86C1;border-radius:20px;padding:2rem 1.8rem;box-shadow:0 6px 20px rgba(46,134,193,0.12);text-align:center;">
                    <div style="font-size:2.2rem;margin-bottom:0.6rem;">🔑</div>
                    <div style="font-size:1.15rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">Sign in to your Provider account</div>
                    <div style="font-size:0.9rem;color:#4a7a99;margin-bottom:1rem;">Manage your service listings and see contact requests from seniors.</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                if known_providers:
                    prov_login_name = st.selectbox("Select your organization", known_providers, key="prov_login_select")
                    if st.button("🔑  Sign In as Provider", use_container_width=True, key="do_prov_login"):
                        st.session_state.provider_logged_in = True
                        st.session_state.provider_name_demo = prov_login_name
                        st.rerun()
                else:
                    st.info("No provider accounts yet. Use **Register as Provider** to create one.")
                    if st.button("✨  Go to Register", use_container_width=True, key="goto_reg"):
                        st.session_state.prov_tab_mode = "register"
                        st.rerun()

            else:  # register
                st.markdown("""
                <div style="background:#ffffff;border:2px solid #27AE60;border-radius:20px;padding:2rem 1.8rem;box-shadow:0 6px 20px rgba(39,174,96,0.1);text-align:center;">
                    <div style="font-size:2.2rem;margin-bottom:0.6rem;">🏥</div>
                    <div style="font-size:1.15rem;font-weight:800;color:#1A4B6E;margin-bottom:0.3rem;">Register your organization</div>
                    <div style="font-size:0.9rem;color:#4a7a99;">Free to register. Reach seniors who are already looking for services like yours.</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                new_prov_name = st.text_input("Organization name", placeholder="e.g. Victoria Meals Society", key="prov_reg_name")
                new_prov_type = st.selectbox("Type of organization", ["Non-profit", "Health authority", "Government", "Private provider", "Community group"], key="prov_reg_type")
                if st.button("✅  Register & Sign In", use_container_width=True, key="do_prov_reg"):
                    if new_prov_name.strip():
                        if "registered_providers" not in st.session_state:
                            st.session_state.registered_providers = {}
                        st.session_state.registered_providers[new_prov_name.strip()] = {"type": new_prov_type}
                        st.session_state.provider_logged_in = True
                        st.session_state.provider_name_demo = new_prov_name.strip()
                        st.rerun()
                    else:
                        st.warning("Please enter your organization name.")

        st.markdown("<br>", unsafe_allow_html=True)
        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">📋</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">List Your Services</div><div style="font-size:0.9rem;color:#4a7a99;">Add and manage your service listings so seniors can find you easily.</div></div>""", unsafe_allow_html=True)
        with pf2:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">📨</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Receive Requests</div><div style="font-size:0.9rem;color:#4a7a99;">See when seniors want to contact you and respond directly.</div></div>""", unsafe_allow_html=True)
        with pf3:
            st.markdown("""<div class="section-card" style="text-align:center;"><div style="font-size:2rem;">✅</div><div style="font-weight:800;color:#1A4B6E;margin:0.4rem 0;">Keep Info Current</div><div style="font-size:0.9rem;color:#4a7a99;">Update your hours, fees, and contact details whenever things change.</div></div>""", unsafe_allow_html=True)

    else:
        # ── LOGGED IN: header + sign out
        prov_hdr1, prov_hdr2 = st.columns([3, 1])
        with prov_hdr1:
            st.markdown(f"""
            <div class="assistant-box" style="display:flex;align-items:center;gap:1rem;">
                <div style="font-size:2.5rem;">🏢</div>
                <div>
                    <div style="font-size:1.2rem;font-weight:800;color:#1A4B6E;">Welcome, {st.session_state.provider_name_demo}</div>
                    <div style="font-size:0.95rem;color:#4a7a99;">Provider account · Subscription: Active (demo)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with prov_hdr2:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("🚪  Sign Out", use_container_width=True, key="prov_signout"):
                st.session_state.provider_logged_in = False
                st.session_state.provider_name_demo = ""
                st.session_state.prov_tab_mode = "login"
                st.rerun()

        provider_subtab1, provider_subtab2, provider_subtab3 = st.tabs([
            "➕  Add a Service",
            "📋  My Service Listings",
            "📨  Contact Requests"
        ])

        with provider_subtab1:
            st.markdown("""
            <div style="background:#ffffff;border:1.5px solid #D6E8F5;border-left:5px solid #2E86C1;border-radius:14px;padding:0.9rem 1.2rem;margin-bottom:1rem;">
                <div style="font-size:1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.2rem;">➕ Add a New Service</div>
                <div style="font-size:0.88rem;color:#4a7a99;">Fill in the details below. Use plain language that seniors will understand easily.</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("provider_add_service_form"):
                p1, p2 = st.columns(2)
                with p1:
                    service_name = st.text_input("Service name *", placeholder="e.g. Free Meals Delivery")
                    organization = st.text_input("Organization name *", value=st.session_state.provider_name_demo)
                    contact_person = st.text_input("Contact person name")
                    category = st.selectbox("Service category *", [
                        "Health & Home Care", "Food & Nutrition", "Transportation",
                        "Housing", "Social Activities", "Income Support",
                        "Mobility", "Emergency", "Recreation", "Education",
                        "Computer Services", "Taxes & Administrative Support"
                    ])
                    p_location = st.text_input("City", value="Victoria")
                    area_of_victoria = st.text_input("Area of Victoria", placeholder="e.g. Saanich, Oak Bay, Downtown")
                    address = st.text_input("Physical address", placeholder="e.g. 123 Oak Bay Ave, Victoria BC")

                with p2:
                    fee_type = st.selectbox("Cost to seniors", ["Free", "Partial pay", "Subscription", "Private pay"])
                    access_mode = st.selectbox("How to access", ["In person", "Online", "Both"])
                    transit_access = st.selectbox("Transit accessible?", ["Yes", "No", "Limited"])
                    phone = st.text_input("Phone number")
                    email = st.text_input("Email address")
                    website = st.text_input("Website", placeholder="https://...")
                    communication_options = st.multiselect("Also available via", ["Chat", "FAQ"])
                    eligibility = st.text_input("Who can use this service?", placeholder="e.g. Seniors 65 and older in Victoria")

                description = st.text_area(
                    "Plain-language description *",
                    placeholder="Describe what your service does in simple words a senior would understand. Example: 'We deliver hot meals to your door every weekday so you don't have to cook.'"
                )

                st.markdown("<div style='font-size:0.88rem;color:#4a7a99;margin-top:0.5rem;'>How would seniors rate this service?</div>", unsafe_allow_html=True)
                r1, r2, r3, r4, r5 = st.columns(5)
                with r1: rating = st.slider("Overall", 0, 5, 4)
                with r2: friendly_score = st.slider("Friendly", 0, 5, 4)
                with r3: efficient_score = st.slider("Efficient", 0, 5, 4)
                with r4: easy_to_understand_score = st.slider("Clear", 0, 5, 4)
                with r5: accessibility_score = st.slider("Accessible", 0, 5, 4)

                submitted = st.form_submit_button("✅  Save Service Listing", use_container_width=True)
                if submitted:
                    if service_name.strip() and description.strip():
                        add_provider_service(
                            service_name=service_name.strip(),
                            category=category,
                            subcategory="",
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
                        st.success(f"✅ '{service_name}' has been added to the directory. Seniors can now find it.")
                        st.rerun()
                    else:
                        st.warning("Service name and description are required.")

        with provider_subtab2:
            # Show all services in the directory that belong to this provider
            prov_name = st.session_state.provider_name_demo
            my_services = st.session_state.services_df[
                st.session_state.services_df["Organization"].str.lower().str.contains(prov_name.lower(), na=False) |
                st.session_state.services_df["Provider"].str.lower().str.contains(prov_name.lower(), na=False)
            ].copy()

            st.markdown(f"""
            <div style="background:#ffffff;border:1.5px solid #D6E8F5;border-left:5px solid #2E86C1;border-radius:14px;padding:0.9rem 1.2rem;margin-bottom:1rem;">
                <div style="font-size:1rem;font-weight:800;color:#1A4B6E;">📋 Your Service Listings</div>
                <div style="font-size:0.88rem;color:#4a7a99;">{len(my_services)} service{"s" if len(my_services) != 1 else ""} currently listed under <strong>{prov_name}</strong>.</div>
            </div>
            """, unsafe_allow_html=True)

            if my_services.empty:
                st.info("No services listed yet. Use the 'Add a Service' tab to add your first listing.")
            else:
                for _, row in my_services.iterrows():
                    fee = row.get("FeeType", "")
                    fee_color = "#27AE60" if fee == "Free" else "#E67E22" if fee == "Partial pay" else "#8E44AD" if fee == "Subscription" else "#E74C3C"
                    _phone = str(row.get("Phone", "")).strip()
                    _website = str(row.get("Website", "")).strip()
                    _address = str(row.get("Address", "")).strip()
                    maps_html = ""
                    if _address and _address not in ["nan", ""] and "Online" not in _address:
                        maps_q = _address.replace(" ", "+")
                        maps_html = f'&nbsp;·&nbsp; <a href="https://www.google.com/maps/search/?api=1&query={maps_q}" target="_blank" style="color:#E67E22;font-weight:600;text-decoration:none;font-size:0.82rem;">🗺️ Map</a>'
                    web_html = ""
                    if _website and _website not in ["nan", ""]:
                        web_html = f'&nbsp;·&nbsp; <a href="{_website}" target="_blank" style="color:#2E86C1;font-weight:600;text-decoration:none;font-size:0.82rem;">🌐 Website</a>'
                    st.markdown(
                        f'<div class="service-card" style="border-left-color:{fee_color};">'
                        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.3rem;">'
                        f'<div class="service-title" style="font-size:1.05rem;">{row["ServiceName"]}</div>'
                        f'<span style="background:{fee_color}22;color:{fee_color};border:1.5px solid {fee_color}55;border-radius:999px;padding:0.15rem 0.8rem;font-size:0.82rem;font-weight:800;">{fee}</span>'
                        f'</div>'
                        f'<div class="service-meta" style="margin-bottom:0.4rem;">{row["FriendlyCategory"]} &nbsp;·&nbsp; {row.get("AreaOfVictoria","Victoria")} &nbsp;·&nbsp; {row.get("AccessMode","In person")}</div>'
                        f'<div style="font-size:0.92rem;color:#2d5a7a;line-height:1.65;margin-bottom:0.4rem;">{row["Description"]}</div>'
                        f'<div style="font-size:0.85rem;color:#4a7a99;">'
                        f'{"📞 " + _phone if _phone and _phone not in ["nan",""] else ""}'
                        f'{maps_html}{web_html}'
                        f'</div>'
                        f'<div style="margin-top:0.4rem;font-size:0.8rem;color:#4a7a99;">{verification_label(row)}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with provider_subtab3:
            st.markdown("""
            <div style="background:#ffffff;border:1.5px solid #D6E8F5;border-left:5px solid #27AE60;border-radius:14px;padding:0.9rem 1.2rem;margin-bottom:1rem;">
                <div style="font-size:1rem;font-weight:800;color:#1A4B6E;">📨 Incoming Contact Requests</div>
                <div style="font-size:0.88rem;color:#4a7a99;">These are seniors or caregivers who want to connect with your organization.</div>
            </div>
            """, unsafe_allow_html=True)
            provider_requests = st.session_state.user_requests.copy()
            provider_requests = provider_requests[
                provider_requests["Provider"].str.lower() == st.session_state.provider_name_demo.lower()
            ]
            if provider_requests.empty:
                st.info("No contact requests yet. Once seniors find your service and send a message, it will appear here.")
            else:
                for _, req in provider_requests.iterrows():
                    status_color = "#27AE60" if req["Status"] == "Replied" else "#E67E22" if req["Status"] == "Pending" else "#4a7a99"
                    st.markdown(f"""
                    <div class="service-card">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                            <div class="service-title" style="font-size:1.05rem;">{req['ServiceName']}</div>
                            <span style="background:{status_color}22;color:{status_color};border:1.5px solid {status_color}55;border-radius:999px;padding:0.2rem 0.8rem;font-size:0.82rem;font-weight:700;">{req['Status']}</span>
                        </div>
                        <div style="font-size:0.9rem;color:#4a7a99;margin-bottom:0.5rem;">From: <strong>{req['UserName']}</strong> ({req['UserRole']}) · Contact by: {req['PreferredContact']} · Received: {req['CreatedAt']}</div>
                        <div style="font-size:0.95rem;color:#2d5a7a;background:#F8FBFF;border-radius:10px;padding:0.7rem 0.9rem;">{req['Message']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    new_status = st.selectbox(
                        "Update status", ["Pending", "Replied", "Closed"],
                        index=["Pending", "Replied", "Closed"].index(req["Status"]),
                        key=f"status_{req['RequestID']}"
                    )
                    if st.button("Save", key=f"save_status_{req['RequestID']}", use_container_width=True):
                        st.session_state.user_requests.loc[
                            st.session_state.user_requests["RequestID"] == req["RequestID"], "Status"
                        ] = new_status
                        st.success("Status updated.")
                        st.rerun()

# ---------------- HELP & FAQ ----------------
with tab5:

    # ── Big friendly header
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A4B6E 0%,#2E86C1 100%);border-radius:20px;padding:1.8rem 2rem;margin-bottom:1.6rem;box-shadow:0 8px 24px rgba(26,75,110,0.2);">
        <div style="font-size:2rem;font-weight:800;color:white;margin-bottom:0.4rem;">❓ Help &amp; Common Questions</div>
        <div style="font-size:1.05rem;color:rgba(255,255,255,0.82);line-height:1.7;">Pick a topic below. Tap any question to see the answer in plain, simple language.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── EMERGENCY NUMBERS — always visible at top
    st.markdown("""
    <div style="background:#FFF5F5;border:2px solid #E74C3C;border-radius:18px;padding:1.2rem 1.5rem;margin-bottom:1.4rem;">
        <div style="font-size:1.1rem;font-weight:800;color:#C0392B;margin-bottom:0.7rem;">🆘 Important Phone Numbers — Always Free to Call</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;">
            <div style="background:#fff;border-radius:12px;padding:0.9rem 1rem;text-align:center;border:1.5px solid #FADBD8;">
                <div style="font-size:1.6rem;font-weight:800;color:#C0392B;">2-1-1</div>
                <div style="font-size:0.88rem;font-weight:700;color:#1A4B6E;margin-top:2px;">BC211 — Any Service Help</div>
                <div style="font-size:0.8rem;color:#4a7a99;margin-top:2px;">24 hours · Free · Any language</div>
            </div>
            <div style="background:#fff;border-radius:12px;padding:0.9rem 1rem;text-align:center;border:1.5px solid #FADBD8;">
                <div style="font-size:1.6rem;font-weight:800;color:#C0392B;">310-6789</div>
                <div style="font-size:0.88rem;font-weight:700;color:#1A4B6E;margin-top:2px;">Mental Health Support</div>
                <div style="font-size:0.8rem;color:#4a7a99;margin-top:2px;">24 hours · Free · No area code</div>
            </div>
            <div style="background:#fff;border-radius:12px;padding:0.9rem 1rem;text-align:center;border:1.5px solid #FADBD8;">
                <div style="font-size:1.6rem;font-weight:800;color:#C0392B;">1-866-437-1940</div>
                <div style="font-size:0.88rem;font-weight:700;color:#1A4B6E;margin-top:2px;">Senior Abuse Support</div>
                <div style="font-size:0.8rem;color:#4a7a99;margin-top:2px;">Free · Confidential</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Topic picker — big visual buttons instead of dropdown
    if "faq_topic" not in st.session_state:
        st.session_state.faq_topic = None

    st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#1A4B6E;margin-bottom:0.8rem;">What do you need help with? Tap a topic:</div>', unsafe_allow_html=True)

    topic_items = [
        ("🔍", "Using This Directory",       "#EBF5FB", "#2E86C1"),
        ("🍽️", "Food &amp; Meals",           "#E8F8F2", "#27AE60"),
        ("🚌", "Getting Around",              "#FEF9E7", "#E67E22"),
        ("🏠", "Housing Help",               "#F5F0FF", "#8E44AD"),
        ("🏥", "Health &amp; Home Care",     "#EBF5FB", "#2980B9"),
        ("💰", "Money &amp; Benefits",       "#F0FFF4", "#16A085"),
        ("💻", "Computer &amp; Phone Help",  "#F0FFF4", "#1E8449"),
        ("🆘", "Safety &amp; Emergency",     "#FFF5F5", "#C0392B"),
    ]

    t_cols = st.columns(4)
    for i, (icon, label, bg, color) in enumerate(topic_items):
        with t_cols[i % 4]:
            clean_label = label.replace("&amp;", "&")
            is_active = st.session_state.faq_topic == clean_label
            if st.button(
                f"{icon}  {clean_label}",
                key=f"faq_topic_{i}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.faq_topic = None if is_active else clean_label
                st.rerun()

    # ── FAQ answers — show when a topic is selected
    FAQ_CONTENT = {
        "Using This Directory": [
            ("🔍  How do I search for a service?",
             "Three easy ways:\n\n**1. Type it** — write what you need in plain words. Try: *\"I need help with meals\"* or *\"I need a doctor\"*\n\n**2. Tap a button** — on the Find Services page, tap one of the ready-made buttons like \"I need food help\" or \"I need transportation\"\n\n**3. Speak it** — press the microphone button and say your question out loud. No typing needed."),
            ("💲  Is this free to use?",
             "**Yes — completely free.** You can search and browse without making an account. Making an account (also free) lets you save services and send messages to providers. No password is ever needed — just your name."),
            ("👤  Do I need to create an account?",
             "**No account needed to search.** Just go to Find Services and start searching. If you want to save a service or contact a provider through the app, tap My Account and enter your name — that's all."),
            ("💾  How do I save a service I like?",
             "1. Sign in to My Account — just enter your name (free)\n2. Search for the service you want\n3. Tap **Save This Service** on the service card\n\nYour saved services will always be waiting for you in My Account."),
            ("📞  How do I contact a provider?",
             "Every service card shows the phone number, email, and website. You can call or email directly — no account needed. If you are signed in, you can also send a message through the app and the provider will call or email you back."),
        ],
        "Food & Meals": [
            ("🍽️  Can I get meals delivered to my home?",
             "**Yes — Meals on Wheels Victoria** delivers hot meals every day to seniors 60 and older who have difficulty cooking. It is **free**.\n\n📞 Call: **250-386-6313**\n\nVolunteer Grocery Delivery is also available for seniors who cannot leave home."),
            ("🥗  Where can I get a free meal?",
             "**Community Meals Program** serves a free hot lunch Monday to Friday in a friendly community setting. The **Greater Victoria Food Bank** provides free food hampers — no questions asked.\n\nCall **2-1-1** to find the nearest location to you."),
            ("📋  How do I sign up for Meals on Wheels?",
             "Just call **250-386-6313**. A friendly staff member will ask a few simple questions and set up delivery. No paperwork — just a phone call."),
        ],
        "Getting Around": [
            ("🚌  What is HandyDART?",
             "**HandyDART** is a door-to-door bus service for seniors and people with mobility challenges. A bus picks you up at your door and takes you where you need to go.\n\nYou must book in advance.\n\n📞 Call: **250-727-7811**"),
            ("🚕  What is the Taxi Saver Program?",
             "The **Taxi Saver Program** gives you subsidized taxi vouchers. You buy the vouchers at a discount and use them for taxi rides to medical appointments and errands.\n\n📞 Call BC Transit: **250-727-7811** to apply."),
            ("🎟️  Is there a cheaper bus pass for seniors?",
             "Yes — the **BC Bus Pass Program** offers a reduced-price monthly transit pass for low-income seniors 65 and older.\n\n📞 Call: **1-866-866-0800**\n\nOr apply online at **gov.bc.ca**"),
            ("🚗  Can I get a free ride to a medical appointment?",
             "Yes — the **Volunteer Driver Program** provides free rides to medical appointments. Trained volunteers come to your door.\n\n📞 Call Volunteer Victoria: **250-477-3535**"),
        ],
        "Housing Help": [
            ("🏠  What if I can't afford my rent?",
             "The **SAFER Program** (Shelter Aid For Elderly Renters) gives monthly money to help pay rent for low-income seniors 60 and older. The payment goes directly to your landlord.\n\n📞 Call BC Housing: **1-800-257-7756**\n🌐 Website: **bchousing.org**"),
            ("🔨  Can I get help making my home safer?",
             "Yes — BC Housing's **Home Repair Program** gives free grants to low-income homeowners 60 and older for safety upgrades like grab bars, ramps, and better lighting.\n\n📞 Call: **1-800-257-7756**"),
            ("🏡  What is assisted living?",
             "Assisted living is a home where you have your own space but get help with meals, housekeeping, and personal care. It is for seniors who want independence but need some daily support.\n\n📞 Call Island Health: **250-370-8699** to learn if you qualify."),
        ],
        "Health & Home Care": [
            ("🏠  How do I get care help at home?",
             "Call **Island Health** to ask for a home care assessment. A nurse will visit to understand what help you need. If you qualify, you can get free help with bathing, dressing, meals, and nursing visits — in your own home.\n\n📞 Call: **250-370-8699**"),
            ("😮‍💨  What is respite care?",
             "**Respite care** gives family caregivers a much-needed break. A trained care worker looks after your loved one for a few hours or days so you can rest.\n\n📞 Call Island Health: **250-370-8699**"),
            ("🧠  Is there mental health support for seniors?",
             "**Yes — free, 24 hours a day, 7 days a week.**\n\nCall **310-6789** (no area code). Trained counsellors provide phone support for anyone feeling stressed, anxious, or overwhelmed."),
            ("🦷  Is there affordable dental care?",
             "Yes — **Island Health Dental** provides subsidized dental care for low-income seniors 65 and older — checkups, cleaning, and treatments.\n\n📞 Call: **250-388-6328**"),
        ],
        "Money & Benefits": [
            ("💰  What money help is available for seniors?",
             "Several programs can help:\n\n- **SAFER** — rent help (call 1-800-257-7756)\n- **BC Income Assistance** — monthly living support (call 1-866-866-0800)\n- **Energy Bill Assistance** — BC Hydro help (call 1-866-866-0800)\n- **BC Bus Pass** — cheaper transit (call 1-866-866-0800)\n- **Free Tax Help** — your taxes done for free (call 250-386-3393)\n\nNot sure which one is for you? Call **2-1-1** for free guidance."),
            ("📝  Can I get help filing my taxes for free?",
             "**Yes — completely free.** The CRA Volunteer Tax Program has trained volunteers who complete your tax return at no cost.\n\n📞 Call: **250-386-3393**\n🌐 Website: **canada.ca/cra-volunteer-tax**"),
            ("💡  How do I get help with my energy bill?",
             "**BC Hydro** offers a Low Income Energy Assistance Program for qualifying seniors — a monthly credit on your bill.\n\n📞 Call: **1-866-866-0800**\n🌐 Website: **bchydro.com/low-income-program**"),
        ],
        "Computer & Phone Help": [
            ("💻  I have never used a computer. Where do I start?",
             "**Victoria Public Library** offers free computer classes for complete beginners. Patient instructors teach you step by step at your own pace. **No experience needed at all.**\n\n📍 Visit: **735 Broughton St, Victoria** (the main library)\n📞 Call: **250-384-0613**"),
            ("📱  Can I get help with my phone or tablet?",
             "**Yes — walk in, no appointment needed.**\n\nThe **Library Digital Drop-in** at Victoria Public Library (735 Broughton St) provides free one-on-one help with smartphones, tablets, and computers.\n\n📞 Call: **250-413-4700**"),
            ("📧  How do I learn to use email and the internet?",
             "**SeniorsBC.ca/digital** has free online guides and videos made just for seniors — email, internet safety, video calls, online banking, and more.\n\nOr visit the library for in-person help."),
            ("⚠️  How do I stay safe from scams online?",
             "Never give your password, banking details, or personal information to anyone who contacts you unexpectedly.\n\nThe library digital help staff can show you how to recognize scams. If you have been targeted, call the **Senior Abuse Support Line: 1-866-437-1940** — free and confidential."),
        ],
        "Safety & Emergency": [
            ("🆘  What if I am being abused or taken advantage of?",
             "**Call the Senior Abuse Support Line: 1-866-437-1940**\n\nFree. Confidential. Available to all seniors. Staff will help you understand your options and connect you with free legal advice if needed.\n\nFor immediate danger, **call 9-1-1**."),
            ("🔔  What is a personal emergency alert?",
             "A **personal emergency alert** is a small button you wear — on your wrist or around your neck. When you press it, it immediately connects you to help — at home or anywhere.\n\n📞 Call Lifeline Canada: **1-800-387-8122** for information."),
            ("🩹  What if I fall at home and can't reach the phone?",
             "A **personal emergency alert button** means you can get help even if you can't reach a phone.\n\nYou can also ask a neighbour or family member to check on you regularly.\n\nCall **2-1-1** to learn about programs that help keep seniors safe at home."),
        ],
    }

    if st.session_state.faq_topic:
        topic_key = st.session_state.faq_topic
        questions = FAQ_CONTENT.get(topic_key, [])
        if questions:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#EFF6FF,#F6F9FF);border:1.5px solid #C8DDF0;border-left:5px solid #2778B5;border-radius:14px;padding:0.9rem 1.3rem;margin:1rem 0 0.8rem;">
                <div style="font-size:1.1rem;font-weight:800;color:#0F2D45;">Questions about: {topic_key}</div>
                <div style="font-size:0.88rem;color:#5a7d99;margin-top:2px;">Tap any question to read the answer.</div>
            </div>
            """, unsafe_allow_html=True)
            for q, a in questions:
                with st.expander(f"  {q}"):
                    import re as _re
                    # Convert **text** → <strong>text</strong> for proper HTML rendering
                    a_html = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', a)
                    # Convert \n → <br> for line breaks
                    a_html = a_html.replace('\n\n', '<br><br>').replace('\n', '<br>')
                    st.markdown(
                        '<div style="background:#F8FBFF;border-left:4px solid #2778B5;border-radius:0 12px 12px 0;'
                        'padding:1.3rem 1.5rem;font-size:1.08rem;color:#1e3a50;line-height:2.0;">'
                        + a_html +
                        '</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.markdown("""
        <div style="background:#F8FBFF;border:1.5px dashed #AED6F1;border-radius:14px;padding:1.5rem;text-align:center;margin-top:0.5rem;">
            <div style="font-size:1.3rem;margin-bottom:0.4rem;">👆</div>
            <div style="font-size:1rem;font-weight:700;color:#1A4B6E;">Tap one of the topics above to see answers.</div>
            <div style="font-size:0.9rem;color:#4a7a99;margin-top:0.3rem;">Or call BC211 (dial 2-1-1) any time — free, 24 hours, speaks many languages.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#EBF5FB;border:1.5px solid #AED6F1;border-radius:14px;padding:1rem 1.3rem;">
        <div style="font-size:0.95rem;font-weight:800;color:#1A4B6E;margin-bottom:0.4rem;">💻 Free Computer &amp; Phone Help at the Victoria Library</div>
        <div style="font-size:0.92rem;color:#2d5a7a;line-height:1.75;">Walk in — no appointment needed. Staff will sit with you one-on-one and help with your phone, tablet, or computer at no cost.<br><strong>📍 735 Broughton St, Victoria &nbsp; · &nbsp; 📞 250-413-4700</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PROVIDER FAQ SECTION
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A5276 0%,#27AE60 100%);border-radius:18px;padding:1.4rem 1.8rem;margin-bottom:1.2rem;box-shadow:0 6px 20px rgba(26,82,118,0.18);">
        <div style="font-size:1.5rem;font-weight:800;color:white;margin-bottom:0.3rem;">🏥 For Service Providers</div>
        <div style="font-size:0.95rem;color:rgba(255,255,255,0.82);">Are you a clinic, non-profit, health authority, or community organization? Here's what you need to know.</div>
    </div>
    """, unsafe_allow_html=True)

    prov_faqs = [
        ("🏥  How do I list my organization's services on this directory?",
         "Go to the **For Providers** tab at the top of this page. Click **Register as Provider**, enter your organization name, and you are in. Then go to **Add a Service** and fill in the details.\n\nIt takes about 5 minutes to list your first service."),
        ("💰  What does it cost to list our services?",
         "**Free listing:** You can list up to 3 services at no cost. Your services appear in search results with basic contact information.\n\n**Provider Subscriber ($49–$99/month):** Unlimited listings, priority placement in search, receive contact requests from seniors directly through the app, update info anytime, and see ratings and feedback.\n\nSeniors always use the directory for free — the subscription cost is only for providers."),
        ("✏️  How do I update our service information?",
         "Sign in to the **For Providers** tab, go to **My Service Listings**, and you will see all your current services. You can edit hours, fees, eligibility, phone numbers, and descriptions anytime. Changes take effect immediately — no waiting for approval."),
        ("📨  How do seniors contact us through the app?",
         "When a senior finds your service and is signed into their account, they can tap **Contact Provider** on your service card and send you a message. The message appears in your **Contact Requests** tab in the Provider Portal. You can see their name, preferred contact method, and their question.\n\nYou then contact them directly by phone or email."),
        ("📋  Can we list more than one service?",
         "**Yes — absolutely.** One organization can list as many services as they offer. For example, Island Health can list their Home Care program, Adult Day Program, Dental Care, and Respite Care all under the same provider account.\n\nEach service gets its own card in the directory. When a senior searches, they see all your services that match their need."),
        ("⭐  What are the ratings and how do they work?",
         "Seniors who have used a service can leave a rating (1–5 stars) in four areas: **Friendly**, **Efficient**, **Easy to Understand**, and **Accessible**. These ratings appear on your service card.\n\nAs a subscriber, you can see your ratings and read the feedback. This helps you understand what seniors experience and where you can improve."),
        ("📊  What is the Analytics Dashboard in the Regional License?",
         "The Analytics Dashboard (available in the Regional License tier) shows you:\n\n- Which services seniors search for most\n- Which categories have gaps — services seniors need but can't find\n- How often your services appear in search results\n- Contact request trends over time\n- Geographic patterns — which neighbourhoods search the most\n\nThis data helps CCWS, health authorities, and funders understand where senior service gaps exist in Greater Victoria."),
        ("🏛️  What is the Regional License — who is it for?",
         "The Regional License is for large organizations — like Island Health, the City of Victoria, or the Capital Regional District — who want to run their own version of this platform.\n\nWith a Regional License, you get:\n- **Your own branding** — your logo, your colours, your website address (e.g. victoria.seniorsservices.ca)\n- **White-label deployment** — seniors see it as your platform, not CCWS's\n- **Your own data** — you own the directory, providers, and service records\n- **API connection** — links to your existing systems like referral databases or booking systems\n\nIn plain terms: instead of listing on CCWS's directory, you run your own version of it for your whole region."),
    ]

    for pq, pa in prov_faqs:
        with st.expander(f"{pq}"):
            import re as _re2
            pa_html = _re2.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', pa)
            pa_html = pa_html.replace('\n\n', '<br><br>').replace('\n', '<br>')
            st.markdown(
                '<div style="background:#F0F7FF;border-left:4px solid #27AE60;border-radius:0 12px 12px 0;'
                'padding:1.2rem 1.5rem;font-size:1.05rem;color:#1e3a50;line-height:1.95;">'
                + pa_html + '</div>',
                unsafe_allow_html=True
            )

# ---------------- Business Model & Financial Viability ----------------
with tab6:

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
            <div style="font-size:0.82rem;color:#1A5632;font-weight:700;margin-bottom:0.7rem;padding:0.5rem 0.8rem;background:#C8EDD9;border-radius:8px;">What does "Regional License" mean? → A health authority, municipality, or large non-profit pays once per year to deploy this platform with <strong>their own name and logo</strong> for their entire region — like "Victoria Seniors Directory powered by Island Health." They own the data and can customize everything.</div>
            <ul style="color:#2d5a7a;font-size:0.95rem;line-height:2.1;padding-left:1.1rem;">
                <li><strong>White-label</strong> — your logo, your colours, your domain</li>
                <li>Deploy for any region of BC</li>
                <li>Unlimited provider listings</li>
                <li>Full analytics dashboard</li>
                <li>API connection to your existing systems</li>
                <li>Dedicated setup and support</li>
            </ul>
            <div style="margin-top:1rem;padding:0.75rem;background:#E8F8F2;border-radius:10px;font-size:0.88rem;color:#1A5632;font-weight:600;">
                🏛️ For municipalities, health regions &amp; large non-profits
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