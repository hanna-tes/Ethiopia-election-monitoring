import pandas as pd
import numpy as np
import re
import logging
import time
from datetime import timedelta
from itertools import combinations
import streamlit as st
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import os
import shutil
from collections import defaultdict, Counter
import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io
import base64

# --- Clear Streamlit Cache on Startup ---
def clear_streamlit_cache():
    cache_dir = ".streamlit/cache"
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
        except: pass
clear_streamlit_cache()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config: Ethiopia Lexicon with Category-Term Mapping ---
CONFIG = {
    "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
    "bertrend": {"min_cluster_size": 3},
    "analysis": {"time_window": "48H"},
    "coordination_detection": {"threshold": 0.85, "max_features": 5000},
    
    # === ETHIOPIA LEXICON: Category-Term Mapping ===
    # Structure: {category: {term: {severity, target_entity, language}}}
    "lexicon": {
        # === Ethnic/Identity-Based Terms ===
        "ethnic_identity": {
            # Amhara-related terms
            "አማራ": {"severity": "medium", "target_entity": "Amhara", "language": "amharic"},
            "amhara": {"severity": "medium", "target_entity": "Amhara", "language": "english"},
            "ነፍጠኛ": {"severity": "high", "target_entity": "Amhara", "language": "amharic"},
            "neftegna": {"severity": "high", "target_entity": "Amhara", "language": "english"},
            
            # Oromo-related terms
            "ኦሮሞ": {"severity": "medium", "target_entity": "Oromo", "language": "amharic"},
            "oromo": {"severity": "medium", "target_entity": "Oromo", "language": "english"},
            "ጋላ": {"severity": "high", "target_entity": "Oromo", "language": "amharic"},
            "galla": {"severity": "high", "target_entity": "Oromo", "language": "english"},
            
            # Tigrayan-related terms
            "ትግሬ": {"severity": "medium", "target_entity": "Tigrayan", "language": "amharic"},
            "tigrayan": {"severity": "medium", "target_entity": "Tigrayan", "language": "english"},
            "ወያኔ": {"severity": "high", "target_entity": "TPLF", "language": "amharic"},
            "woyane": {"severity": "high", "target_entity": "TPLF", "language": "english"},
            "ህወሓት": {"severity": "high", "target_entity": "TPLF", "language": "amharic"},
            "tplf": {"severity": "high", "target_entity": "TPLF", "language": "english"},
            
            # Other ethnic groups
            "ቅማንት": {"severity": "medium", "target_entity": "Qemant", "language": "amharic"},
            "qemant": {"severity": "medium", "target_entity": "Qemant", "language": "english"},
            "አገው": {"severity": "medium", "target_entity": "Agew", "language": "amharic"},
            "agew": {"severity": "medium", "target_entity": "Agew", "language": "english"},
            "ሶማሌ": {"severity": "medium", "target_entity": "Somali", "language": "amharic"},
            "አፋር": {"severity": "medium", "target_entity": "Afar", "language": "amharic"},
        },
        
        # === Political Groups & Parties ===
        "political_groups": {
            "ብልፅግና": {"severity": "low", "target_entity": "Prosperity Party", "language": "amharic"},
            "prosperity party": {"severity": "low", "target_entity": "Prosperity Party", "language": "english"},
            "አዴፓ": {"severity": "low", "target_entity": "ADP", "language": "amharic"},
            "adp": {"severity": "low", "target_entity": "ADP", "language": "english"},
            "ፋኖ": {"severity": "medium", "target_entity": "Fano", "language": "amharic"},
            "fano": {"severity": "medium", "target_entity": "Fano", "language": "english"},
            "ኦነግ": {"severity": "high", "target_entity": "ONEG", "language": "amharic"},
            "oneg": {"severity": "high", "target_entity": "ONEG", "language": "english"},
        },
        
        # === Violence & Incitement Terms ===
        "violence_incitement": {
            "ግደል": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "kill": {"severity": "critical", "target_entity": "", "language": "english"},
            "ግደሉ": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "kill them": {"severity": "critical", "target_entity": "", "language": "english"},
            "አጥፋ": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "destroy": {"severity": "critical", "target_entity": "", "language": "english"},
            "ጦርነት": {"severity": "high", "target_entity": "", "language": "amharic"},
            "war": {"severity": "high", "target_entity": "", "language": "english"},
            "ጥቃት": {"severity": "high", "target_entity": "", "language": "amharic"},
            "attack": {"severity": "high", "target_entity": "", "language": "english"},
            "ስጋት": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "threat": {"severity": "medium", "target_entity": "", "language": "english"},
        },
        
        # === Dehumanizing & Derogatory Terms ===
        "dehumanizing": {
            "እንስሳ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "animal": {"severity": "high", "target_entity": "", "language": "english"},
            "ከብት": {"severity": "high", "target_entity": "", "language": "amharic"},
            "cattle": {"severity": "high", "target_entity": "", "language": "english"},
            "ውሻ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "dog": {"severity": "high", "target_entity": "", "language": "english"},
            "ደደብ": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "fool": {"severity": "medium", "target_entity": "", "language": "english"},
            "ቆሻሻ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "trash": {"severity": "high", "target_entity": "", "language": "english"},
            "ሌባ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "thief": {"severity": "high", "target_entity": "", "language": "english"},
            "ገዳይ": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "killer": {"severity": "critical", "target_entity": "", "language": "english"},
        },
        
        # === Election & Governance Terms ===
        "election_governance": {
            "ምርጫ": {"severity": "low", "target_entity": "", "language": "amharic"},
            "election": {"severity": "low", "target_entity": "", "language": "english"},
            "ድምፅ": {"severity": "low", "target_entity": "", "language": "amharic"},
            "vote": {"severity": "low", "target_entity": "", "language": "english"},
            "ነቤ": {"severity": "low", "target_entity": "NEBE", "language": "amharic"},
            "nebe": {"severity": "low", "target_entity": "NEBE", "language": "english"},
            "የተጭበረበረ": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "rigged": {"severity": "medium", "target_entity": "", "language": "english"},
            "ማጭበርበር": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "fraud": {"severity": "medium", "target_entity": "", "language": "english"},
        },
        
        # === Foreign Interference & Geopolitics ===
        "foreign_interference": {
            "ግብፅ": {"severity": "low", "target_entity": "Egypt", "language": "amharic"},
            "egypt": {"severity": "low", "target_entity": "Egypt", "language": "english"},
            "ሱዳን": {"severity": "low", "target_entity": "Sudan", "language": "amharic"},
            "sudan": {"severity": "low", "target_entity": "Sudan", "language": "english"},
            "ኤርትራ": {"severity": "low", "target_entity": "Eritrea", "language": "amharic"},
            "eritrea": {"severity": "low", "target_entity": "Eritrea", "language": "english"},
            "አሜሪካ": {"severity": "low", "target_entity": "USA", "language": "amharic"},
            "america": {"severity": "low", "target_entity": "USA", "language": "english"},
            "ቻይና": {"severity": "low", "target_entity": "China", "language": "amharic"},
            "china": {"severity": "low", "target_entity": "China", "language": "english"},
            "ውጭ": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "foreign": {"severity": "medium", "target_entity": "", "language": "english"},
        },
        
        # === Religious & Cultural Terms ===
        "religious_cultural": {
            "ኦርቶዶክስ": {"severity": "low", "target_entity": "Orthodox", "language": "amharic"},
            "orthodox": {"severity": "low", "target_entity": "Orthodox", "language": "english"},
            "እስልምና": {"severity": "low", "target_entity": "Islam", "language": "amharic"},
            "islam": {"severity": "low", "target_entity": "Islam", "language": "english"},
            "ክርስቲያን": {"severity": "low", "target_entity": "Christian", "language": "amharic"},
            "christian": {"severity": "low", "target_entity": "Christian", "language": "english"},
        }
    },
    
    # === Risk Scoring Configuration ===
    "risk_scoring": {
        "severity_weights": {"low": 1, "medium": 2, "high": 3, "critical": 4},
        "category_weights": {
            "ethnic_identity": 1.2, "political_groups": 1.2, "violence_incitement": 1.5,
            "dehumanizing": 1.5, "election_governance": 1.0, "foreign_interference": 1.0, "religious_cultural": 1.0
        },
        "risk_thresholds": {"low": 3, "medium": 6, "high": 10, "critical": 15}
    },
    
    # === Display Configuration ===
    "display": {"max_terms_per_category": 20, "show_amharic_first": True, "highlight_critical": True}
}

# --- Groq Setup ---
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    if GROQ_API_KEY:
        import groq
        client = groq.Groq(api_key=GROQ_API_KEY)
    else:
        client = None
except Exception as e:
    logger.warning(f"Groq client setup failed: {e}")
    client = None

# --- URLs ---
CFA_LOGO_URL = "https://opportunities.codeforafrica.org/wp-content/uploads/sites/5/2015/11/1-Zq7KnTAeKjBf6eENRsacSQ.png"
MELTWATER_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/MeltwaterEthiopiaMar8.csv?token=GHSAT0AAAAAADRDAPFL7BRYK6DIX4HFMBUQ2PF6EBQ"
CIVICSIGNALS_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaCivicsignalMar8.csv?token=GHSAT0AAAAAADRDAPFLHLGT4462WCZI6VWC2PF6APQ"
TIKTOK_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaTikTokApril.csv?token=GHSAT0AAAAAADRDAPFKFLXKMHU7Y6GFMYXS2PF7K6Q"
OPENMEASURES_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaopenmeasuresApri17.csv?token=GHSAT0AAAAAADRDAPFLUMOFJ7I22WQ7EJ6S2PF7JPA"
ORIGINAL_POSTS_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaMeltwaterApril17Original%20-%20Sheet1.csv?token=GHSAT0AAAAAADRDAPFKEGAQGEFNLB5HVP6W2PGKDXA"

# --- Helper Functions ---
def load_data_robustly(url, name, default_sep=','):
    df = pd.DataFrame()
    if not url: return df
    attempts = [(',', 'utf-8'), (',', 'utf-8-sig'), ('\t', 'utf-8'), (';', 'utf-8'), ('\t', 'utf-16'), (',', 'latin-1')]
    for sep, enc in attempts:
        try:
            df = pd.read_csv(url, sep=sep, low_memory=False, on_bad_lines='skip', encoding=enc)
            if not df.empty and len(df.columns) > 1:
                logger.info(f"✅ {name} loaded (Shape: {df.shape})")
                return df
        except: pass
    logger.error(f"❌ {name} failed to load")
    return pd.DataFrame()

def safe_llm_call(prompt, max_tokens=2048):
    if client is None: return None
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=max_tokens
        )
        try: return response.choices[0].message.content.strip()
        except: return str(response).strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

def infer_platform_from_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"): return "Unknown"
    url = url.lower()
    platforms = {
        "tiktok.com": "TikTok", "vt.tiktok.com": "TikTok", "facebook.com": "Facebook", "fb.watch": "Facebook",
        "twitter.com": "X", "x.com": "X", "youtube.com": "YouTube", "youtu.be": "YouTube",
        "instagram.com": "Instagram", "telegram.me": "Telegram", "t.me": "Telegram", "telegram.org": "Telegram"
    }
    for key, val in platforms.items():
        if key in url: return val
    if any(d in url for d in ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "aljazeera.com"]):
        return "News/Media"
    return "Media"

def extract_original_text(text):
    if pd.isna(text) or not isinstance(text, str): return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt|repost|shared|via|credit)\s*[:@]\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+|http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip().lower()

def is_original_post(text):
    if pd.isna(text) or not isinstance(text, str): return False
    lower = text.strip().lower()
    if not lower: return False
    patterns = [r'^🔁.*reposted', r'\b(reposted|reshared|retweeted)\b', r'^(rt|qt|repost)\s*[:@\s]', r'^\s*[🔁↪️➡️]\s*@?\w*']
    if any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns): return False
    if len(re.sub(r'http\S+|\@\w+', '', text).strip()) < 15: return False
    return len(lower) >= 20 and not re.search(r'^\s*["\u201c]|\s*@\w+\s*[":]', lower)

def parse_timestamp_robust(timestamp):
    if pd.isna(timestamp): return pd.NaT
    ts_str = re.sub(r'\s+GMT$', '', str(timestamp).strip(), flags=re.IGNORECASE)
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True)
        if pd.notna(parsed): return parsed
    except: pass
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M', '%b %d, %Y %H:%M', '%Y-%m-%d']:
        try:
            parsed = pd.to_datetime(ts_str, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed): return parsed
        except: continue
    return pd.NaT

# --- Combine Datasets ---
def combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df=None, openmeasures_df=None):
    combined = []
    def get_col(df, cols):
        df_cols = [c.lower().strip() for c in df.columns]
        for col in cols:
            norm = col.lower().strip()
            if norm in df_cols: return df[df.columns[df_cols.index(norm)]]
        return pd.Series([np.nan]*len(df), index=df.index)
    
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id', 'id'])
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'headline', 'text', 'content'])
        mw['URL'] = get_col(meltwater_df, ['url'])
        mw['timestamp_share'] = get_col(meltwater_df, ['date', 'timestamp', 'alternate date format'])
        mw['source_dataset'] = 'Meltwater'
        combined.append(mw)
    
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'author', 'username'])
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'post_id', 'id'])
        cs['object_id'] = get_col(civicsignals_df, ['title', 'text', 'content', 'body'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'timestamp', 'date'])
        cs['source_dataset'] = 'Civicsignal'
        combined.append(cs)
    
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['object_id'] = get_col(tiktok_df, ['text', 'Transcript', 'caption', 'content'])
        tt['account_id'] = get_col(tiktok_df, ['authorMeta.name', 'username', 'creator'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'itemId'])
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'TikTok Link', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date'])
        tt['source_dataset'] = 'TikTok'
        combined.append(tt)
    
    if openmeasures_df is not None and not openmeasures_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasures_df, ['context_name'])
        om['content_id'] = get_col(openmeasures_df, ['id'])
        om['object_id'] = get_col(openmeasures_df, ['text'])
        om['URL'] = get_col(openmeasures_df, ['url'])
        om['timestamp_share'] = get_col(openmeasures_df, ['created_at'])
        om['source_dataset'] = 'OpenMeasure'
        combined.append(om)
    
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()

def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    if df.empty:
        return pd.DataFrame(columns=['account_id','content_id','object_id','URL','timestamp_share','Platform','original_text','Outlet','Channel','cluster','source_dataset','Sentiment'])
    
    dfp = df.copy()
    if 'Sentiment' in dfp.columns:
        dfp = dfp[dfp['Sentiment'].isin(['Negative', 'Neutral'])]
    if 'object_id' in dfp.columns:
        mask = dfp['object_id'].apply(is_original_post) & (~dfp['object_id'].str.contains('🔁', na=False)) & (~dfp['object_id'].str.startswith('RT @', na=False))
        dfp = dfp[mask].copy()
    
    dfp['object_id'] = dfp['object_id'].astype(str).replace('nan','').fillna('')
    dfp = dfp[dfp['object_id'].str.strip() != ""]
    dfp['original_text'] = dfp['object_id'].apply(extract_original_text) if coordination_mode=="Text Content" else dfp['URL'].astype(str).replace('nan','')
    dfp = dfp[dfp['original_text'].str.strip() != ""].reset_index(drop=True)
    dfp['Platform'] = dfp['URL'].apply(infer_platform_from_url)
    
    if 'source_dataset' in dfp.columns:
        for p in ['TikTok','tiktok','vt.tiktok']: dfp.loc[dfp['source_dataset'].str.contains(p, case=False, na=False), 'Platform'] = 'TikTok'
        for p in ['Telegram','telegram','t.me']: dfp.loc[dfp['source_dataset'].str.contains(p, case=False, na=False), 'Platform'] = 'Telegram'
        for p in ['Media','News','Civicsignal']: dfp.loc[dfp['source_dataset'].str.contains(p, case=False, na=False), 'Platform'] = 'Media'
    
    dfp['Outlet'], dfp['Channel'], dfp['cluster'] = np.nan, np.nan, -1
    if 'Sentiment' not in dfp.columns: dfp['Sentiment'] = np.nan
    cols = ['account_id','content_id','object_id','URL','timestamp_share','Platform','original_text','Outlet','Channel','cluster','source_dataset','Sentiment']
    return dfp[[c for c in cols if c in dfp.columns]].copy()

@st.cache_data(show_spinner=False)
def cached_clustering(df, eps, min_samples, max_features):
    if df.empty or 'original_text' not in df.columns: return df
    df_filt = df[df['original_text'].str.len() > 15].copy()
    if df_filt.empty: return df
    try:
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(3,5), max_features=max_features).fit_transform(df_filt['original_text'])
        clusters = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(tfidf)
        df_out = df.copy(); df_out['cluster'] = -1
        df_out.loc[df_filt.index, 'cluster'] = clusters
        return df_out
    except: return df

def assign_virality_tier(n):
    if n>=500: return "Tier 4: Viral Emergency"
    elif n>=100: return "Tier 3: High Spread"
    elif n>=20: return "Tier 2: Moderate"
    else: return "Tier 1: Limited"

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

# === LEXICON MANAGEMENT FUNCTIONS ===
def get_lexicon_as_dataframe():
    """Convert CONFIG lexicon to DataFrame for display/editing"""
    rows = []
    lexicon = CONFIG.get("lexicon", {})
    for category, terms in lexicon.items():
        for term, metadata in terms.items():
            rows.append({
                'term': term,
                'category': category,
                'severity': metadata.get('severity', 'medium'),
                'target_entity': metadata.get('target_entity', ''),
                'language': metadata.get('language', 'english')
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['term', 'category', 'severity', 'target_entity', 'language'])

def add_term_to_lexicon(term, category, severity, target_entity, language):
    """Add a new term to CONFIG lexicon"""
    term_clean = term.strip().lower()
    if not term_clean:
        return False, "Term cannot be empty"
    
    lexicon = CONFIG.get("lexicon", {})
    if category not in lexicon:
        lexicon[category] = {}
    
    if term_clean in lexicon[category]:
        return False, "Term already exists in this category"
    
    lexicon[category][term_clean] = {
        'severity': severity,
        'target_entity': target_entity.strip() if target_entity else '',
        'language': language
    }
    return True, "Term added successfully"

def update_term_in_lexicon(old_term, old_category, new_term, new_category, severity, target_entity, language):
    """Update an existing term in CONFIG lexicon"""
    lexicon = CONFIG.get("lexicon", {})
    
    # Remove old term
    if old_category in lexicon and old_term in lexicon[old_category]:
        del lexicon[old_category][old_term]
    
    # Add updated term
    if new_category not in lexicon:
        lexicon[new_category] = {}
    
    lexicon[new_category][new_term.strip().lower()] = {
        'severity': severity,
        'target_entity': target_entity.strip() if target_entity else '',
        'language': language
    }
    return True, "Term updated successfully"

def delete_term_from_lexicon(term, category):
    """Delete a term from CONFIG lexicon"""
    lexicon = CONFIG.get("lexicon", {})
    if category in lexicon and term in lexicon[category]:
        del lexicon[category][term]
        return True, "Term deleted successfully"
    return False, "Term not found"

def export_lexicon_to_csv():
    """Export lexicon to CSV format"""
    df = get_lexicon_as_dataframe()
    return df.to_csv(index=False).encode('utf-8')

def import_lexicon_from_csv(uploaded_file):
    """Import lexicon from CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['term', 'category', 'severity', 'language']
        if not all(col in df.columns for col in required_cols):
            return False, f"Missing required columns: {required_cols}"
        
        imported = 0
        for _, row in df.iterrows():
            term = str(row['term']).strip().lower()
            category = str(row['category']).strip()
            severity = str(row.get('severity', 'medium')).strip().lower()
            target_entity = str(row.get('target_entity', '')).strip()
            language = str(row.get('language', 'english')).strip().lower()
            
            if term and category and severity in ['low', 'medium', 'high', 'critical'] and language in ['amharic', 'english']:
                add_term_to_lexicon(term, category, severity, target_entity, language)
                imported += 1
        
        return True, f"Successfully imported {imported} terms"
    except Exception as e:
        return False, f"Import failed: {str(e)}"

def scan_text_for_lexicon_terms(text, category_filter=None):
    """Scan text for lexicon matches using CONFIG mapping"""
    if not isinstance(text, str) or not text.strip():
        return []
    
    text_lower = text.lower()
    matches = []
    lexicon = CONFIG.get("lexicon", {})
    categories_to_check = category_filter if category_filter else lexicon.keys()
    
    for category in categories_to_check:
        if category not in lexicon: continue
        for term, metadata in lexicon[category].items():
            if metadata.get("language") == "amharic" or re.match(r'^[\u1200-\u137F]+$', term):
                pattern = re.escape(term)
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
            
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append({
                    'term': term, 'category': category,
                    'severity': metadata.get('severity', 'medium'),
                    'target_entity': metadata.get('target_entity', ''),
                    'language': metadata.get('language', 'english')
                })
    return matches

def calculate_risk_score(matches):
    """Calculate risk score based on matched terms"""
    if not matches:
        return {'score': 0, 'level': 'low', 'breakdown': {}, 'term_count': 0}
    
    scoring = CONFIG.get("risk_scoring", {})
    severity_weights = scoring.get("severity_weights", {'low': 1, 'medium': 2, 'high': 3, 'critical': 4})
    category_weights = scoring.get("category_weights", {})
    thresholds = scoring.get("risk_thresholds", {'low': 3, 'medium': 6, 'high': 10, 'critical': 15})
    
    total_score = 0
    breakdown = defaultdict(int)
    
    for match in matches:
        sev = match.get('severity', 'medium')
        cat = match.get('category', 'general')
        weight = severity_weights.get(sev, 2) * category_weights.get(cat, 1.0)
        total_score += weight
        breakdown[cat] += weight
    
    if total_score >= thresholds.get('critical', 15): level = 'critical'
    elif total_score >= thresholds.get('high', 10): level = 'high'
    elif total_score >= thresholds.get('medium', 6): level = 'medium'
    else: level = 'low'
    
    return {'score': round(total_score, 2), 'level': level, 'breakdown': dict(breakdown), 'term_count': len(matches)}

def generate_lexicon_analytics(filtered_df, category_filter=None):
    """Generate analytics from lexicon matches"""
    analytics = {
        'total_posts_scanned': len(filtered_df), 'posts_with_matches': 0, 'total_matches': 0,
        'category_distribution': defaultdict(int), 'severity_distribution': defaultdict(int),
        'top_terms': Counter(), 'top_targeted_entities': Counter(), 'risk_level_distribution': defaultdict(int),
        'platform_breakdown': defaultdict(lambda: defaultdict(int)), 'temporal_trend': defaultdict(int), 'language_distribution': defaultdict(int)
    }
    
    if filtered_df.empty or 'original_text' not in filtered_df.columns:
        return analytics
    
    for _, row in filtered_df.iterrows():
        text = str(row.get('original_text', ''))
        matches = scan_text_for_lexicon_terms(text, category_filter)
        
        if matches:
            analytics['posts_with_matches'] += 1
            analytics['total_matches'] += len(matches)
            risk = calculate_risk_score(matches)
            analytics['risk_level_distribution'][risk['level']] += 1
            
            for match in matches:
                analytics['category_distribution'][match['category']] += 1
                analytics['severity_distribution'][match['severity']] += 1
                analytics['top_terms'][match['term']] += 1
                analytics['language_distribution'][match['language']] += 1
                if match['target_entity']:
                    analytics['top_targeted_entities'][match['target_entity']] += 1
                platform = str(row.get('Platform', 'Unknown'))
                analytics['platform_breakdown'][platform][match['category']] += 1
            
            ts = row.get('timestamp_share')
            if pd.notna(ts):
                date_key = pd.Timestamp(ts).strftime('%Y-%m-%d')
                analytics['temporal_trend'][date_key] += len(matches)
    
    for key in ['category_distribution', 'severity_distribution', 'risk_level_distribution', 'temporal_trend', 'language_distribution']:
        analytics[key] = dict(analytics[key])
    analytics['platform_breakdown'] = {k: dict(v) for k, v in analytics['platform_breakdown'].items()}
    return analytics

# === WORD CLOUD FUNCTIONS ===
def generate_trigger_wordcloud(dataset_triggers, width=1000, height=500):
    if not dataset_triggers: return None
    term_freq = {}
    for category, terms in dataset_triggers.items():
        for item in terms:
            term_freq[item['term']] = item['count']
    if not term_freq: return None
    
    wordcloud = WordCloud(
        width=width, height=height, background_color='white', max_words=100,
        colormap='Blues', contour_width=1, contour_color='steelblue',
        stopwords=set(STOPWORDS), prefer_horizontal=0.7, min_font_size=8,
        max_font_size=80, relative_scaling=0.5, normalize_plurals=True
    ).generate_from_frequencies(term_freq)
    return wordcloud

def wordcloud_to_base64(wordcloud_obj):
    img_buffer = io.BytesIO()
    wordcloud_obj.to_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode()

# --- Professional UI Theme ---
def inject_custom_css():
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
        .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #1e293b; font-weight: 600; }
        .stTabs [data-baseweb="tab"] { 
            background: white; border-radius: 8px 8px 0 0; 
            padding: 0.8rem 1.2rem; font-weight: 500; color: #475569;
        }
        .stTabs [aria-selected="true"] { 
            background: #3b82f6; color: white !important; 
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        .metric-card { 
            background: white; border-radius: 12px; padding: 1.2rem; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 4px solid #3b82f6;
        }
        .dataframe { border-radius: 8px; overflow: hidden; }
        .stButton>button { 
            background: #3b82f6; color: white; border: none; 
            border-radius: 6px; padding: 0.5rem 1.2rem; font-weight: 500;
        }
        .stButton>button:hover { background: #2563eb; }
        .amharic { font-family: 'Nyala', 'Kefa', 'Noto Sans Ethiopic', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="🇪🇹 Ethiopia Election Monitor", page_icon="🗳️")
    
    # Header
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image(CFA_LOGO_URL, width=100)
    with col_title:
        st.markdown("## 🇪🇹 Ethiopia Election Monitoring Dashboard")
        st.caption("Dataset-driven triggers • Amharic support • Lexicon management")
    
    # --- Data Loading ---
    with st.spinner("📥 Loading Ethiopia election data..."):
        meltwater_df = load_data_robustly(MELTWATER_URL, "Meltwater")
        civicsignals_df = load_data_robustly(CIVICSIGNALS_URL, "Civicsignal")
        tiktok_df = load_data_robustly(TIKTOK_URL, "TikTok")
        openmeasures_df = load_data_robustly(OPENMEASURES_URL, "OpenMeasure")
    
    combined_raw = combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df, openmeasures_df)
    
    if combined_raw.empty:
        st.error("❌ No data loaded. Please check URLs or upload files manually.")
        st.stop()
    
    # Preprocess
    df_full = final_preprocess_and_map_columns(combined_raw)
    df_full['timestamp_share'] = df_full['timestamp_share'].apply(parse_timestamp_robust)
    
    # Date filter
    valid_dates = df_full['timestamp_share'].dropna()
    if valid_dates.empty:
        st.error("❌ No valid timestamps found.")
        st.stop()
    
    min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    selected_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    start_date = pd.Timestamp(selected_range[0] if len(selected_range)==2 else selected_range[0], tz='UTC')
    end_date = (pd.Timestamp(selected_range[1], tz='UTC') + pd.Timedelta(days=1)) if len(selected_range)==2 else start_date + pd.Timedelta(days=1)
    
    filtered_df = df_full[(df_full['timestamp_share'] >= start_date) & (df_full['timestamp_share'] < end_date)].copy()
    
    # Clustering
    df_clustered = cached_clustering(filtered_df, eps=0.3, min_samples=2, max_features=5000) if not filtered_df.empty else pd.DataFrame()
    
    # Metrics
    total_posts = len(filtered_df)
    unique_accounts = filtered_df['account_id'].nunique()
    top_platform = filtered_df['Platform'].mode()[0] if not filtered_df['Platform'].mode().empty else "—"
    clusters_found = len(df_clustered[df_clustered['cluster'] != -1]['cluster'].unique()) if not df_clustered.empty else 0
    
    # === TABS ===
    tabs = st.tabs([
        "🏠 Dashboard", "📊 Insights", "🔍 Coordination", "⚠️ Risk", "📰 Narratives",
        "🕸️ Network Intelligence", "🎯 Triggers & Entities"
    ])
    
    # === TAB 1: Dashboard ===
    with tabs[0]:
        st.markdown("### 📈 Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Posts Analyzed", f"{total_posts:,}")
        m2.metric("Unique Accounts", f"{unique_accounts:,}")
        m3.metric("Top Platform", top_platform)
        m4.metric("Coordination Clusters", clusters_found)
        st.divider()
        if not filtered_df.empty:
            plot_df = filtered_df.set_index('timestamp_share').resample('H').size()
            if not plot_df.empty:
                fig = px.area(plot_df, title="Hourly Post Volume", labels={'value': 'Posts'})
                st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 2: Insights ===
    with tabs[1]:
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(filtered_df['account_id'].value_counts().head(10), title="Top Accounts")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(filtered_df['Platform'].value_counts(), title="Platform Distribution")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered_df[['timestamp_share', 'Platform', 'account_id', 'object_id']].head(10), use_container_width=True)
    
    # === TAB 3: Coordination ===
    with tabs[2]:
        if not df_clustered.empty and 'cluster' in df_clustered.columns:
            coord = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').filter(lambda x: len(x['account_id'].unique()) > 1)
            if not coord.empty:
                st.success(f"✅ Found {coord['cluster'].nunique()} coordination groups")
                for cid, group in coord.groupby('cluster'):
                    with st.expander(f"🔗 Cluster {cid} — {len(group)} posts, {group['account_id'].nunique()} accounts"):
                        st.code(group['original_text'].iloc[0][:200] + "...", language=None)
                        st.dataframe(group[['timestamp_share', 'account_id', 'Platform', 'URL']].head(5), use_container_width=True)
            else:
                st.info("ℹ️ No multi-account coordination detected.")
        else:
            st.info("ℹ️ Upload more data to detect coordination.")
    
    # === TAB 4: Risk ===
    with tabs[3]:
        st.markdown("### ⚠️ Narrative Risk Overview")
        if not df_clustered.empty:
            sizes = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').size()
            if not sizes.empty:
                risk_df = pd.DataFrame({'Cluster': sizes.index, 'Count': sizes.values, 'Virality': sizes.apply(assign_virality_tier)})
                fig = px.bar(risk_df.nlargest(10, 'Count'), x='Cluster', y='Count', color='Virality', title="Top Clusters by Volume")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(risk_df.nlargest(10, 'Count'), use_container_width=True)
    
    # === TAB 5: Narratives ===
    with tabs[4]:
        st.markdown("### 📰 Trending Narratives")
        st.info("ℹ️ Narrative summaries require LLM API key. Showing raw cluster data instead.")
        if not df_clustered.empty:
            for cid in df_clustered[df_clustered['cluster'] != -1]['cluster'].unique()[:5]:
                cluster_posts = df_clustered[df_clustered['cluster'] == cid]
                st.markdown(f"#### Cluster #{cid} — {len(cluster_posts)} posts")
                st.write(cluster_posts['original_text'].iloc[0][:300] + "...")
                st.caption(f"Platforms: {', '.join(cluster_posts['Platform'].unique())}")
                st.divider()
    
    # === TAB 6: Network & Coordination Intelligence ===
    with tabs[5]:
        st.subheader("🕸️ Network & Coordination Intelligence")
        if not df_clustered.empty and 'cluster' in df_clustered.columns:
            G = nx.Graph()
            for cid, group in df_clustered[df_clustered['cluster'] != -1].groupby('cluster'):
                accounts = group['account_id'].dropna().unique()
                if len(accounts) > 1:
                    for i in range(len(accounts)):
                        for j in range(i+1, len(accounts)):
                            G.add_edge(accounts[i], accounts[j], weight=1)
            
            if G.number_of_edges() > 0:
                pos = nx.spring_layout(G, seed=42)
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#94a3b8'), hoverinfo='skip'))
                node_x = [pos[n][0] for n in G.nodes()]
                node_y = [pos[n][1] for n in G.nodes()]
                node_deg = [G.degree(n) for n in G.nodes()]
                
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode='markers+text',
                    text=[str(n)[:12] for n in G.nodes()], textposition="bottom center",
                    marker=dict(size=[d*3+8 for d in node_deg], color=node_deg, colorscale='Viridis'),
                    hoverinfo='text', text=[f"{n}<br>Degree: {G.degree(n)}" for n in G.nodes()]
                ))
                fig.update_layout(title="Account Coordination Network", height=500, plot_bgcolor='white',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Nodes", G.number_of_nodes())
                col2.metric("Edges", G.number_of_edges())
                col3.metric("Avg Degree", f"{sum(dict(G.degree()).values())/G.number_of_nodes():.1f}")
                
                if st.button("📥 Export Network"):
                    st.download_button("Download JSON", json.dumps(nx.node_link_data(G)), "network.json", "application/json")
            else:
                st.info("ℹ️ No coordination links detected.")
        else:
            st.info("ℹ️ Upload data to generate network.")
    
    # === TAB 7: Triggers & Entities WITH LEXICON MANAGEMENT ===
    with tabs[6]:
        st.subheader("🎯 Trigger Terms & Targeted Entities")
        st.markdown("*Ethiopia Hate Speech Lexicon • Amharic/English Support • Non-technical Management*")
        
        # 4 sub-tabs including Lexicon Management
        sub_tabs = st.tabs(["🔍 Trigger Scanner", "👥 Entity Registry", "📊 Analytics", "⚙️ Lexicon Management"])
        
        # --- Sub-tab 1: Trigger Scanner ---
        with sub_tabs[0]:
            st.markdown("### 🔍 Real-Time Trigger Detection")
            lexicon_df = get_lexicon_as_dataframe()
            total_terms = len(lexicon_df)
            st.info(f"✅ Lexicon contains {total_terms:,} terms across {len(CONFIG['lexicon'])} categories")
            
            # Test text input
            test_text = st.text_area("Paste text to scan for hate speech terms", height=100, placeholder="Example: የወያኔ ድስት ላሺ...")
            if test_text and st.button("🔎 Scan Text"):
                matches = scan_text_for_lexicon_terms(test_text)
                risk = calculate_risk_score(matches)
                if matches:
                    st.error(f"🚨 Found {len(matches)} hate speech term(s)")
                    risk_color = {'low': '✅', 'medium': '⚠️', 'high': '🔴', 'critical': '🚨'}
                    st.markdown(f"**Risk Level:** {risk_color.get(risk['level'], '⚪')} {risk['level'].upper()} (Score: {risk['score']})")
                    st.dataframe(pd.DataFrame(matches)[['term', 'category', 'severity', 'target_entity']], use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No lexicon matches found")
            
            st.divider()
            
            # Scan dataset
            if st.button("🔎 Scan All Filtered Posts"):
                with st.spinner("Scanning posts..."):
                    flagged = []
                    for _, row in filtered_df.iterrows():
                        text = str(row.get('original_text', ''))
                        matches = scan_text_for_lexicon_terms(text)
                        if matches:
                            risk = calculate_risk_score(matches)
                            flagged.append({
                                'account_id': row.get('account_id'), 'platform': row.get('Platform'),
                                'timestamp': row.get('timestamp_share'), 'matched_terms': [m['term'] for m in matches],
                                'categories': list(set([m['category'] for m in matches])),
                                'risk_level': risk['level'], 'risk_score': risk['score']
                            })
                    if flagged:
                        flag_df = pd.DataFrame(flagged)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Flagged Posts", len(flag_df))
                        m2.metric("High/Critical Risk", len(flag_df[flag_df['risk_level'].isin(['high', 'critical'])]))
                        m3.metric("Unique Accounts", flag_df['account_id'].nunique())
                        st.dataframe(flag_df[['timestamp', 'platform', 'account_id', 'matched_terms', 'risk_level']].head(50), use_container_width=True)
                        st.download_button("📥 Download Flagged", convert_df_to_csv(flag_df), "lexicon_flagged.csv", "text/csv")
                    else:
                        st.info("✅ No matches in current filter window.")
        
        # --- Sub-tab 2: Entity Registry ---
        with sub_tabs[1]:
            st.markdown("### 👥 Targeted Entity Registry")
            targeted_entities_df = extract_targeted_entities(filtered_df)
            if not targeted_entities_df.empty:
                st.success(f"✅ Identified {len(targeted_entities_df)} potential targeted entities")
                search = st.text_input("🔍 Search entities (supports Amharic)")
                display_df = targeted_entities_df if not search else targeted_entities_df[targeted_entities_df['entity'].str.contains(search, case=False, na=False, regex=False)]
                st.dataframe(display_df[['entity', 'mention_count', 'mentioned_by', 'platform']].head(30), use_container_width=True, hide_index=True, column_config={"mentioned_by": st.column_config.ListColumn("Mentioned By")})
            else:
                st.info("ℹ️ No entities extracted yet.")
        
        # --- Sub-tab 3: Analytics ---
        with sub_tabs[2]:
            st.markdown("### 📊 Lexicon Analytics")
            analytics = generate_lexicon_analytics(filtered_df)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Lexicon Terms", f"{len(get_lexicon_as_dataframe()):,}")
            col2.metric("Posts with Matches", f"{analytics['posts_with_matches']:,}")
            col3.metric("Total Matches", f"{analytics['total_matches']:,}")
            
            if analytics['category_distribution']:
                cat_df = pd.DataFrame([{'category': cat.replace('_', ' ').title(), 'count': cnt} for cat, cnt in analytics['category_distribution'].items()]).sort_values('count', ascending=False)
                fig = px.pie(cat_df, values='count', names='category', title="Triggers by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            if analytics['temporal_trend']:
                trend_df = pd.DataFrame([{'date': d, 'matches': c} for d, c in analytics['temporal_trend'].items()])
                trend_df['date'] = pd.to_datetime(trend_df['date'])
                fig = px.area(trend_df, x='date', y='matches', title="Daily Trigger Mentions")
                st.plotly_chart(fig, use_container_width=True)
        
        # === ✨ NEW: Sub-tab 4: Lexicon Management ===
        with sub_tabs[3]:
            st.subheader("⚙️ Lexicon Management")
            st.markdown("*Add, edit, or remove hate speech terms — no coding required*")
            
            # Initialize session state for lexicon editing
            if 'lexicon_edit_mode' not in st.session_state:
                st.session_state.lexicon_edit_mode = 'view'  # view, add, edit
                st.session_state.edit_term_data = None
            
            # Action buttons
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.button("➕ Add New Term"):
                    st.session_state.lexicon_edit_mode = 'add'
                    st.session_state.edit_term_data = None
                    st.rerun()
            with col2:
                if st.button("📥 Import CSV"):
                    uploaded = st.file_uploader("Upload lexicon CSV", type=['csv'], key='lex_import')
                    if uploaded:
                        success, msg = import_lexicon_from_csv(uploaded)
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
            with col3:
                csv_data = export_lexicon_to_csv()
                st.download_button("📤 Export CSV", csv_data, "ethiopia_lexicon.csv", "text/csv")
            
            st.divider()
            
            # === ADD/EDIT FORM ===
            if st.session_state.lexicon_edit_mode in ['add', 'edit']:
                st.markdown("#### " + ("➕ Add New Term" if st.session_state.lexicon_edit_mode == 'add' else "✏️ Edit Term"))
                
                # Pre-fill if editing
                if st.session_state.edit_term_data:
                    old_term = st.session_state.edit_term_data['term']
                    old_category = st.session_state.edit_term_data['category']
                    init_term = old_term
                    init_category = old_category
                    init_severity = st.session_state.edit_term_data['severity']
                    init_entity = st.session_state.edit_term_data['target_entity']
                    init_lang = st.session_state.edit_term_data['language']
                else:
                    old_term = old_category = None
                    init_term = init_severity = init_entity = init_lang = ""
                    init_category = "ethnic_identity"
                
                col1, col2 = st.columns(2)
                with col1:
                    new_term = st.text_input("Term (supports Amharic) 🇪🇹", value=init_term, key='edit_term')
                    new_category = st.selectbox("Category", list(CONFIG['lexicon'].keys()), index=list(CONFIG['lexicon'].keys()).index(init_category) if init_category in CONFIG['lexicon'] else 0, key='edit_cat')
                    new_severity = st.selectbox("Severity", ["low", "medium", "high", "critical"], index=["low", "medium", "high", "critical"].index(init_severity) if init_severity in ["low", "medium", "high", "critical"] else 1, key='edit_sev')
                with col2:
                    new_entity = st.text_input("Target Entity (optional)", value=init_entity, key='edit_entity')
                    new_language = st.selectbox("Language", ["amharic", "english"], index=0 if init_lang == "amharic" else 1, key='edit_lang')
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Term"):
                        if not new_term.strip():
                            st.error("❌ Term cannot be empty")
                        else:
                            if st.session_state.lexicon_edit_mode == 'add':
                                success, msg = add_term_to_lexicon(new_term, new_category, new_severity, new_entity, new_language)
                            else:
                                success, msg = update_term_in_lexicon(old_term, old_category, new_term, new_category, new_severity, new_entity, new_language)
                            
                            if success:
                                st.success(f"✅ {msg}")
                                st.session_state.lexicon_edit_mode = 'view'
                                st.session_state.edit_term_data = None
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
                with col2:
                    if st.button("🚫 Cancel"):
                        st.session_state.lexicon_edit_mode = 'view'
                        st.session_state.edit_term_data = None
                        st.rerun()
                
                st.info("💡 Tips: • Use lowercase for consistency • Amharic terms auto-detected • Severity affects risk scoring")
                st.divider()
            
            # === LEXICON TABLE VIEW ===
            else:
                st.markdown("#### 📋 Lexicon Terms")
                lexicon_df = get_lexicon_as_dataframe()
                
                if lexicon_df.empty:
                    st.info("ℹ️ No terms in lexicon. Click 'Add New Term' to get started.")
                else:
                    # Filters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        filter_cat = st.multiselect("Filter by Category", list(CONFIG['lexicon'].keys()), default=[], key='lex_filter_cat')
                    with col2:
                        filter_sev = st.multiselect("Filter by Severity", ["low", "medium", "high", "critical"], default=[], key='lex_filter_sev')
                    with col3:
                        search_term = st.text_input("🔍 Search terms", key='lex_search')
                    
                    # Apply filters
                    filtered_lex = lexicon_df.copy()
                    if filter_cat:
                        filtered_lex = filtered_lex[filtered_lex['category'].isin(filter_cat)]
                    if filter_sev:
                        filtered_lex = filtered_lex[filtered_lex['severity'].isin(filter_sev)]
                    if search_term:
                        filtered_lex = filtered_lex[filtered_lex['term'].str.contains(search_term, case=False, na=False)]
                    
                    st.caption(f"Showing {len(filtered_lex)} of {len(lexicon_df)} terms")
                    
                    # Display table with actions
                    if not filtered_lex.empty:
                        # Add action column
                        def make_actions(row):
                            edit_btn = f"✏️ Edit"
                            delete_btn = f"🗑️ Delete"
                            return f"{edit_btn}|{delete_btn}"
                        
                        filtered_lex['actions'] = filtered_lex.apply(make_actions, axis=1)
                        
                        # Display with custom renderer for actions
                        st.dataframe(
                            filtered_lex[['term', 'category', 'severity', 'target_entity', 'language', 'actions']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "severity": st.column_config.SelectboxColumn("Severity", options=["low", "medium", "high", "critical"]),
                                "actions": st.column_config.TextColumn("Actions", help="Click to edit/delete")
                            }
                        )
                        
                        # Handle row clicks (simplified - in production use st.data_editor or custom components)
                        selected_row = st.selectbox("Select term to edit/delete", filtered_lex['term'].unique(), key='lex_select')
                        if selected_row:
                            row_data = filtered_lex[filtered_lex['term'] == selected_row].iloc[0]
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"✏️ Edit '{selected_row}'"):
                                    st.session_state.lexicon_edit_mode = 'edit'
                                    st.session_state.edit_term_data = row_data.to_dict()
                                    st.rerun()
                            with col2:
                                if st.button(f"🗑️ Delete '{selected_row}'", type="primary"):
                                    if st.checkbox(f"Confirm delete '{selected_row}'?", key='confirm_delete'):
                                        success, msg = delete_term_from_lexicon(row_data['term'], row_data['category'])
                                        if success:
                                            st.success(f"✅ {msg}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ {msg}")
                    
                    # Stats
                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Terms", f"{len(lexicon_df):,}")
                    col2.metric("Categories", f"{len(CONFIG['lexicon'])}")
                    col3.metric("Critical Severity", f"{len(lexicon_df[lexicon_df['severity']=='critical']):,}")
                    col4.metric("Amharic Terms", f"{len(lexicon_df[lexicon_df['language']=='amharic']):,}")
            
            # === IMPORT/EXPORT HELP ===
            with st.expander("📚 CSV Format Guide"):
                st.markdown("""
                **Required columns for import:**
                - `term`: The hate speech term (lowercase, supports Amharic)
                - `category`: One of: ethnic_identity, political_groups, violence_incitement, dehumanizing, election_governance, foreign_interference, religious_cultural
                - `severity`: One of: low, medium, high, critical
                - `language`: One of: amharic, english
                - `target_entity`: (optional) The group/person being targeted
                
                **Example row:**
                ```csv
                term,category,severity,target_entity,language
                ግደል,violence_incitement,critical,,amharic
                killer,dehumanizing,critical,,english
                ```
                """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📥 Export Data")
        if not filtered_df.empty:
            st.download_button("Download Filtered CSV", convert_df_to_csv(filtered_df), "ethiopia_data.csv", "text/csv")
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("Ethiopia Election Monitor v1.2\n\nLexicon management • Amharic support • Word cloud visualization\n\nBuilt with Streamlit • Code for Africa")

if __name__ == "__main__":
    main()
