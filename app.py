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

# Define the data directory path
#DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Update URL variables to local file paths
#MELTWATER_URL = os.path.join(DATA_DIR, "MeltwaterEthiopiaMar8.csv")
#CIVICSIGNALS_URL = os.path.join(DATA_DIR, "EthiopiaCivicsignalMar8.csv")
#TIKTOK_URL = os.path.join(DATA_DIR, "EthiopiaTikTokApril.csv")
#OPENMEASURES_URL = os.path.join(DATA_DIR, "EthiopiaopenmeasuresApri17.csv")
#ORIGINAL_POSTS_URL = os.path.join(DATA_DIR, "EthiopiaMeltwaterApril17Original1.csv")
MELTWATER_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/Ethiopia_NOV2025_Apri2026X.csv"
CIVICSIGNALS_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaCivicsignalApril17.csv"
TIKTOK_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaTikTokApril17.csv"
OPENMEASURES_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaopenmeasuresApri17.csv"
ORIGINAL_POSTS_URL = "https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaMeltwaterApril17Original1.csv"

# --- Helper Functions ---
import requests
import io
import re
from urllib.parse import urlparse, parse_qs, unquote

def parse_github_raw_url(url):
    """
    Parse GitHub raw URL to extract owner, repo, branch, and file path.
    Handles URLs with token parameters.
    
    Returns: dict with owner, repo, branch, path, or None if parsing fails
    """
    if not url or 'githubusercontent.com' not in url:
        return None
    
    try:
        # Remove token parameter if present
        if '?token=' in url:
            url = url.split('?token=')[0]
        
        # Parse URL: https://raw.githubusercontent.com/OWNER/REPO/BRANCH/PATH
        pattern = r'raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)$'
        match = re.match(pattern, url)
        
        if match:
            owner, repo, branch, path = match.groups()
            return {
                'owner': owner,
                'repo': repo,
                'branch': branch,
                'path': unquote(path.strip())  # Decode URL-encoded characters
            }
    except Exception as e:
        logger.warning(f"Failed to parse GitHub URL: {e}")
    
    return None

def load_from_github_api(owner, repo, path, branch='main', token=None):
    """
    Load CSV content from private GitHub repo using GitHub Contents API.
    
    Args:
        owner: GitHub username or org
        repo: Repository name
        path: File path within repo
        branch: Branch name (default: 'main')
        token: GitHub PAT for authentication
    
    Returns:
        str: Raw CSV content, or None if failed
    """
    # GitHub Contents API endpoint
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    
    headers = {
        'Accept': 'application/vnd.github.v3.raw',  # Get raw file content
        'User-Agent': 'Ethiopia-Election-Monitor/1.0'
    }
    
    if token:
        headers['Authorization'] = f'token {token}'
    
    # Add branch parameter
    params = {'ref': branch} if branch else {}
    
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"GitHub API fetch failed for {owner}/{repo}/{path}: {e}")
        if response.status_code == 404:
            logger.error(f"File not found: {path}")
        elif response.status_code == 403:
            logger.error("Access denied - check token permissions")
        return None

def load_data_robustly(url, name, default_sep=','):
    """Load CSV from URL or local path with pandas version compatibility"""
    import requests
    from io import StringIO
    
    df = pd.DataFrame()
    if not url:
        logger.warning(f"⚠️ {name}: No URL/path provided")
        return df
    
    # --- Handle local files ---
    if not url.startswith('http'):
        if os.path.exists(url):
            try:
                return pd.read_csv(url, sep=default_sep, low_memory=False, on_bad_lines='skip')
            except Exception as e:
                logger.error(f"❌ {name} local load failed: {e}")
                return pd.DataFrame()
        logger.error(f"❌ {name}: Local file not found: {url}")
        return pd.DataFrame()
    
    # --- Handle URLs ---
    try:
        # Fetch content via requests (more reliable than pandas direct URL)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        # Try parsing with multiple encodings/separators
        # NOTE: Use default 'c' engine with low_memory=False (pandas ≥2.0 compatible)
        attempts = [
            (',', 'utf-8'),
            (',', 'utf-8-sig'),  # Handles BOM
            ('\t', 'utf-8'),      # Tab-separated
            (';', 'utf-8'),       # European CSVs
            (',', 'latin-1'),     # Fallback encoding
        ]
        
        for sep, enc in attempts:
            try:
                df = pd.read_csv(
                    StringIO(content), 
                    sep=sep, 
                    encoding=enc,
                    low_memory=False,      # ✅ Works with default 'c' engine
                    on_bad_lines='skip',
                    # ❌ REMOVED: engine='python' (incompatible with low_memory in pandas ≥2.0)
                )
                if not df.empty and len(df.columns) > 1:
                    logger.info(f"✅ {name} loaded (Sep: '{sep}', Enc: '{enc}', Shape: {df.shape})")
                    return df
            except pd.errors.ParserError as e:
                logger.debug(f"⚠️ {name} parse failed (sep='{sep}', enc='{enc}'): {e}")
                continue
            except UnicodeDecodeError:
                continue  # Try next encoding
                
        logger.error(f"❌ {name}: Could not parse CSV content after all attempts")
        return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ {name}: Failed to fetch URL - {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ {name}: Unexpected error - {type(e).__name__}: {e}")
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
        if col in df.columns:
            return df[col]
        # Then try normalized match (lowercase, stripped)
        df_cols = [c.lower().strip() for c in df.columns]
        for col in cols:
            norm = col.lower().strip()
            if norm in df_cols:
                return df[df.columns[df_cols.index(norm)]]
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
        # ✅ Match exact CSV column names (note the '/' instead of '.')
        tt['object_id'] = get_col(tiktok_df, ['text', 'Transcript', 'caption', 'content'])
        tt['account_id'] = get_col(tiktok_df, ['authorMeta/name', 'username', 'creator'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'itemId'])
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'TikTok Link', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date', 'createTime'])
        tt['source_dataset'] = 'TikTok'
        
        # ✅ Preserve engagement metrics for dashboard insights
        for col in ['playCount', 'diggCount', 'commentCount', 'shareCount', 'repostCount', 'textLanguage']:
            if col in tiktok_df.columns:
                tt[col] = tiktok_df[col]
                
        # ✅ Preserve first 5 hashtag columns
        for i in range(5):
            hashtag_col = f'hashtags/{i}/name'
            if hashtag_col in tiktok_df.columns:
                tt[f'hashtag_{i}'] = tiktok_df[hashtag_col]
                
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
        # Handle NaN values
        dfp['source_dataset'] = dfp['source_dataset'].fillna('')
        
        # Map TikTok
        tiktok_mask = dfp['source_dataset'].str.contains('TikTok|tiktok|vt.tiktok', case=False, na=False)
        dfp.loc[tiktok_mask, 'Platform'] = 'TikTok'
        
        # ✅ FIX: Map OpenMeasure to Telegram (this was missing!)
        telegram_mask = dfp['source_dataset'].str.contains('Telegram|telegram|t.me|OpenMeasure', case=False, na=False)
        dfp.loc[telegram_mask, 'Platform'] = 'Telegram'
        
        # Map Media/News
        media_mask = dfp['source_dataset'].str.contains('Media|News|Civicsignal', case=False, na=False)
        dfp.loc[media_mask, 'Platform'] = 'Media'
    
    # Fill any remaining unknown platforms
    dfp['Platform'] = dfp['Platform'].replace('', 'Unknown').fillna('Unknown')
    
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
        
# --- Summarize Cluster (Ethiopia-Specific LLM Intelligence Report) ---
def summarize_cluster_ethiopia(texts, urls, cluster_data, min_ts, max_ts):
    """
    Generate a STRICT summary using ONLY content explicitly present in texts.
    NO fabrication of accounts, URLs, engagement metrics, or claims.
    """
    # ✅ Use MORE texts for better context (adjust based on your token budget)
    # Using 80 texts gives good coverage without hitting typical limits
    sample_texts = texts[:80]
    joined = "\n---\n".join([f"[{i+1}] {t}" for i, t in enumerate(sample_texts)])
    
    # Only include REAL URLs from the dataset (max 10 for reference)
    real_urls = [u for u in urls if u and u.startswith('http')][:10]
    url_context = "\nReal source links from dataset:\n" + "\n".join(real_urls) if real_urls else ""
    
    prompt = f"""
You are an intelligence analyst reviewing social media posts about the Ethiopia election.
Your task is to summarize ONLY what is explicitly stated in the provided posts.

**STRICT RULES - DO NOT VIOLATE:**
1. Use ONLY the exact text content provided below. Do NOT invent, assume, or extrapolate.
2. Do NOT create fake account names, URLs, engagement metrics, or timestamps.
3. Do NOT mention specific likes/retweets/views unless explicitly present in the text.
4. If a claim is not directly stated in the provided texts, DO NOT include it.
5. If you cannot find evidence for a category, write "Not explicitly stated in provided posts."

**Provided Posts (verbatim from dataset, {len(sample_texts)} samples shown):**
{joined}

**Real Source Links (from dataset, for reference only):**
{url_context}

**Time Range:** {min_ts} to {max_ts}

**Output Format (use simple text, no markdown headers):**
NARRATIVE THEME: [One short phrase summarizing the dominant topic]

EXPLICIT CLAIMS (quote or closely paraphrase from posts above):
- [Claim 1, with brief context]
- [Claim 2, with brief context]
- [etc.]

TARGETED GROUPS/ENTITIES (only if explicitly named in posts):
- [Group/entity 1]
- [Group/entity 2]

LANGUAGE/TONE OBSERVED: [e.g., accusatory, urgent, informational, etc.]

SAMPLE QUOTES (exact phrases from provided posts, max 5):
1. "[exact quote 1]"
2. "[exact quote 2]"
3. "[exact quote 3]"

DO NOT include: fake accounts, fake URLs, engagement metrics, or claims not in the provided texts.
"""
    
    # ✅ Use higher max_tokens since you have capacity
    response = safe_llm_call(prompt, max_tokens=2048)
    
    if not response:
        # Fallback: show raw sample of posts instead of fake summary
        return f"⚠️ Summary generation failed. Sample posts from cluster:\n" + "\n".join(sample_texts[:10])
    
    # Clean output but preserve strictness
    cleaned = re.sub(r'\*\*.*?\*\*', '', response)  # Remove bold markers
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)  # Remove code blocks
    cleaned = re.sub(r'^[\s\-#*]+|[\s\-#*]+$', '', cleaned, flags=re.MULTILINE)  # Trim markdown
    return cleaned.strip()
    
def get_ethiopia_summaries(df_clustered_all, filtered_df):
    """Generates LLM-powered summaries for top narrative clusters"""
    if df_clustered_all.empty or 'cluster' not in df_clustered_all.columns:
        return []
        
    cluster_sizes = df_clustered_all[df_clustered_all['cluster'] != -1].groupby('cluster').size()
    top_15_clusters = cluster_sizes.nlargest(15).index.tolist()
    all_summaries = []
    
    for cluster_id in top_15_clusters:
        cluster_posts = df_clustered_all[df_clustered_all['cluster'] == cluster_id]
        
        # Use all posts for text aggregation
        all_texts = [t for t in cluster_posts['object_id'].astype(str) if len(t.strip()) > 10]
        if not all_texts: continue
            
        total_reach = len(cluster_posts)
        originators = cluster_posts.sort_values('timestamp_share')['account_id'].dropna().unique().tolist()[:5] or ["Unknown"]
        
        min_ts = cluster_posts['timestamp_share'].min()
        max_ts = cluster_posts['timestamp_share'].max()
        min_ts_str = min_ts.strftime('%Y-%m-%d') if pd.notna(min_ts) else 'N/A'
        max_ts_str = max_ts.strftime('%Y-%m-%d') if pd.notna(max_ts) else 'N/A'
        
        raw_response = summarize_cluster_ethiopia(
            all_texts, 
            cluster_posts['URL'].dropna().unique().tolist(), 
            cluster_posts, 
            min_ts_str, 
            max_ts_str
        )
        
        all_summaries.append({
            "cluster_id": cluster_id,
            "Context": raw_response,
            "Originators": ", ".join([str(a) for a in originators]),
            "Amplifiers_Count": len(cluster_posts['account_id'].dropna().unique()),
            "Total_Reach": total_reach,
            "Emerging Virality": assign_virality_tier(total_reach),
            "Top_Platforms": ", ".join([f"{p} ({c})" for p, c in cluster_posts['Platform'].value_counts().head(3).items()]),
            "Min_TS": min_ts,
            "Max_TS": max_ts,
            "Posts_Data": cluster_posts
        })
    return all_summaries
def get_summaries_for_platform(df_clustered_all, filtered_df_global):
    """
    Generates summaries for Tab 4 using the FULL dataset (including reposts) 
    to capture the entire narrative spread.
    """
    if df_clustered_all.empty or 'cluster' not in df_clustered_all.columns:
        return []
        
    # Cluster sizes based on ALL posts
    cluster_sizes = df_clustered_all[df_clustered_all['cluster'] != -1].groupby('cluster').size()
    top_15_clusters = cluster_sizes.nlargest(15).index.tolist()
    all_summaries = []
    
    for cluster_id in top_15_clusters:
        cluster_posts = df_clustered_all[df_clustered_all['cluster'] == cluster_id]
        
        # Use ALL posts in the cluster for text aggregation (amplification measurement)
        all_texts = cluster_posts['object_id'].astype(str).tolist() 
        all_texts = [t for t in all_texts if len(t.strip()) > 10]
        
        # Calculate full reach and platforms
        total_reach = len(cluster_posts)
        amplifiers = cluster_posts['account_id'].dropna().unique().tolist()
        originators = cluster_posts.sort_values('timestamp_share')['account_id'].dropna().unique().tolist()
        originators = originators[:5] if originators else ["Unknown"]

        if not all_texts:
            continue
            
        min_ts = cluster_posts['timestamp_share'].min()
        max_ts = cluster_posts['timestamp_share'].max()
        min_ts_str = min_ts.strftime('%Y-%m-%d') if pd.notna(min_ts) else 'N/A'
        max_ts_str = max_ts.strftime('%Y-%m-%d') if pd.notna(max_ts) else 'N/A'
        
        raw_response = summarize_cluster_ethiopia(  # ✅ Uses Ethiopia-specific summarizer
            all_texts, 
            cluster_posts['URL'].dropna().unique().tolist(), 
            cluster_posts, 
            min_ts_str, 
            max_ts_str
        )
        
        virality = assign_virality_tier(total_reach)
        platform_dist = cluster_posts['Platform'].value_counts()
        top_platforms = ", ".join([f"{p} ({c})" for p, c in platform_dist.head(3).items()])
        
        all_summaries.append({
            "cluster_id": cluster_id,
            "Context": raw_response,
            "Originators": ", ".join([str(a) for a in originators]),
            "Amplifiers_Count": len(amplifiers),
            "Total_Reach": total_reach,
            "Emerging Virality": virality,
            "Top_Platforms": top_platforms,
            "Min_TS": min_ts,
            "Max_TS": max_ts,
            "Posts_Data": cluster_posts
        })
    return all_summaries
    
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
    
def extract_targeted_entities(df, text_col='original_text', account_col='account_id'):
    """Extract potential PEPs/entities from Ethiopia election context"""
    if df.empty or text_col not in df.columns:
        return pd.DataFrame()
    
    entity_patterns = [
        r'\b(Abiy\s+Ahmed|Prosperity\s+Party|FANO|NEBE|National\s+Election\s+Board)\b',
        r'\b(Amhara|Tigray|Oromo|Somali|Afar|Sidama)\b',
        r'\b([A-Z][a-z]+\s+(Region|Zone|Woreda|Council))\b',
        r'[\u1200-\u137F]{3,}(?:\s+[\u1200-\u137F]{2,}){0,2}',
    ]
    
    entities = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ''))
        account = str(row.get(account_col, ''))
        
        for pattern in entity_patterns:
            candidates = re.findall(pattern, text, re.IGNORECASE)
            for candidate in candidates:
                if isinstance(candidate, tuple):
                    candidate = ' '.join(candidate)
                if len(candidate.strip()) >= 3:
                    entities.append({
                        "entity": candidate.strip(),
                        "mentioned_by": account,
                        "platform": row.get('Platform', 'Unknown'),
                        "timestamp": row.get('timestamp_share'),
                        "context": text[:200]
                    })
    
    if not entities:
        return pd.DataFrame()
    
    entity_df = pd.DataFrame(entities)
    aggregated = entity_df.groupby('entity').agg({
        'mentioned_by': lambda x: list(set(x)),
        'platform': lambda x: list(set(x)),
        'timestamp': 'min',
        'context': 'first'
    }).reset_index()
    aggregated['mention_count'] = entity_df.groupby('entity').size().values
    return aggregated.sort_values('mention_count', ascending=False)

# --- Professional UI Theme (Light/Dark Mode Support) ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* ===== CSS VARIABLES FOR THEME ===== */
        :root {
            /* Light mode defaults */
            --bg-primary: #FFFFFF;
            --bg-secondary: #F8FAFC;
            --bg-tertiary: #F1F5F9;
            --text-primary: #0F172A;
            --text-secondary: #475569;
            --text-muted: #94A3B8;
            --border-color: #E2E8F0;
            --primary: #2563EB;          /* Professional Blue */
            --primary-hover: #1D4ED8;
            --success: #059669;          /* Emerald Green */
            --warning: #D97706;          /* Amber */
            --error: #DC2626;            /* Red */
            --critical: #7C2D12;         /* Deep Red for critical */
            --card-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
            --hover-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        }

        /* Dark mode overrides */
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #0F172A;
                --bg-secondary: #1E293B;
                --bg-tertiary: #334155;
                --text-primary: #F8FAFC;
                --text-secondary: #CBD5E1;
                --text-muted: #64748B;
                --border-color: #475569;
                --primary: #3B82F6;
                --primary-hover: #60A5FA;
                --success: #10B981;
                --warning: #F59E0B;
                --error: #EF4444;
                --critical: #F97316;
                --card-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
                --hover-shadow: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -1px rgba(0,0,0,0.2);
            }
        }

        /* ===== GLOBAL APP STYLES ===== */
        .stApp {
            background-color: var(--bg-secondary);
            color: var(--text-primary);
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 600;
            letter-spacing: -0.025em;
        }
        p, span, div, label {
            color: var(--text-primary);
        }
        .stCaption, .stHelp, small {
            color: var(--text-muted) !important;
        }

        /* ===== CARDS & CONTAINERS ===== */
        .metric-card, .stMetric {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: var(--card-shadow);
            transition: box-shadow 0.2s ease;
        }
        .stMetric:hover {
            box-shadow: var(--hover-shadow);
        }
        .stMetric label {
            color: var(--text-secondary) !important;
            font-size: 0.875rem;
        }
        .stMetric div[data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-weight: 600;
        }

        /* ===== BUTTONS ===== */
        .stButton > button {
            background-color: var(--primary);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.25rem;
            font-weight: 500;
            box-shadow: var(--card-shadow);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: var(--primary-hover);
            box-shadow: var(--hover-shadow);
            transform: translateY(-1px);
        }
        .stButton > button:active {
            transform: translateY(0);
        }
        /* Secondary button style */
        .stButton > button[kind="secondary"] {
            background-color: transparent;
            color: var(--primary) !important;
            border: 1px solid var(--border-color);
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: var(--bg-tertiary);
        }

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab"] {
            background: var(--bg-primary);
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 1rem;
            font-weight: 500;
            color: var(--text-secondary);
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background: var(--bg-primary);
            color: var(--primary) !important;
            border-bottom-color: var(--primary);
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary);
        }

        /* ===== DATAFRAMES & TABLES ===== */
        .stDataFrame {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: var(--card-shadow);
        }
        .stDataFrame th {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-secondary) !important;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color) !important;
        }
        .stDataFrame td {
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }
        .stDataFrame tr:hover {
            background-color: var(--bg-tertiary) !important;
        }

        /* ===== ALERTS & STATUS MESSAGES ===== */
        .stAlert {
            background-color: var(--bg-primary) !important;
            border-left: 4px solid var(--primary);
            border-radius: 0 8px 8px 0;
            color: var(--text-primary) !important;
            box-shadow: var(--card-shadow);
        }
        .stAlert-success { border-left-color: var(--success); }
        .stAlert-warning { border-left-color: var(--warning); }
        .stAlert-error { border-left-color: var(--error); }
        .stAlert-info { border-left-color: var(--primary); }

        /* ===== INPUTS & SELECTORS ===== */
        .stTextInput input, .stSelectbox select, .stMultiselect input {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px;
        }
        .stTextInput input:focus, .stSelectbox select:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        /* ===== CHARTS (Plotly) ===== */
        .plotly-chart svg, .plotly-chart .main-svg {
            background-color: var(--bg-primary) !important;
        }
        .plotly-chart .xtick > text, .plotly-chart .ytick > text {
            fill: var(--text-secondary) !important;
        }
        .plotly-chart .title > text {
            fill: var(--text-primary) !important;
            font-weight: 600;
        }
        .plotly-chart .legend > text {
            fill: var(--text-secondary) !important;
        }

        /* ===== EXPANDERS & SECTIONS ===== */
        .streamlit-expanderHeader {
            background-color: var(--bg-primary) !important;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary) !important;
            font-weight: 500;
        }
        .streamlit-expanderContent {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 8px 8px;
            padding: 1rem;
        }

        /* ===== AMHARIC TEXT SUPPORT ===== */
        .amharic, [lang="am"], [lang="amh"] {
            font-family: 'Nyala', 'Kefa', 'Noto Sans Ethiopic', 'Abyssinica SIL', sans-serif;
            line-height: 1.6;
        }

        /* ===== UTILITY CLASSES ===== */
        .text-muted { color: var(--text-muted); }
        .text-success { color: var(--success); }
        .text-warning { color: var(--warning); }
        .text-error { color: var(--error); }
        .text-critical { color: var(--critical); font-weight: 600; }
        
        .border-light { border: 1px solid var(--border-color); border-radius: 8px; }
        .bg-card { background: var(--bg-primary); border-radius: 10px; padding: 1rem; box-shadow: var(--card-shadow); }
        
        /* ===== SCROLLBAR (subtle) ===== */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    </style>
    """, unsafe_allow_html=True)

# DEBUG: Validate URLs before loading
def validate_url(url, name):
    if url.startswith('http'):
        try:
            import requests
            r = requests.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                logger.info(f"✅ {name} URL valid ({r.status_code})")
                return True
            else:
                logger.error(f"❌ {name} URL returned {r.status_code}: {url[:100]}...")
                return False
        except Exception as e:
            logger.error(f"❌ {name} URL check failed: {e}")
            return False
    return True  # Local paths assumed valid

# Run validation
for name, url in [("Meltwater", MELTWATER_URL), ("Civicsignal", CIVICSIGNALS_URL), 
                  ("TikTok", TIKTOK_URL), ("OpenMeasure", OPENMEASURES_URL),
                  ("OriginalPosts", ORIGINAL_POSTS_URL)]:
    validate_url(url, name)
    
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

        # --- Load ORIGINAL POSTS dataset for coordination analysis ---
    # This is SEPARATE from combined_raw - used exclusively for Tab 3 coordination
    with st.spinner("📥 Loading Original Posts dataset (for coordination analysis)..."):
        original_posts_raw_df = load_data_robustly(ORIGINAL_POSTS_URL, "Original Posts Only")
    
    if not original_posts_raw_df.empty:
        # Map columns using same logic as Meltwater
        def get_col(df, cols):
            df_cols = [c.lower().strip() for c in df.columns]
            for col in cols:
                norm = col.lower().strip()
                if norm in df_cols:
                    return df[df.columns[df_cols.index(norm)]]
            return pd.Series([np.nan]*len(df), index=df.index)
        
        original_posts_df = pd.DataFrame()
        original_posts_df['account_id'] = get_col(original_posts_raw_df, ['influencer'])
        original_posts_df['content_id'] = get_col(original_posts_raw_df, ['tweet id', 'post id', 'id'])
        original_posts_df['object_id'] = get_col(original_posts_raw_df, ['hit sentence', 'opening text', 'headline', 'text', 'content'])
        original_posts_df['URL'] = get_col(original_posts_raw_df, ['url'])
        original_posts_df['timestamp_share'] = get_col(original_posts_raw_df, ['date', 'timestamp', 'alternate date format'])
        original_posts_df['source_dataset'] = 'OriginalPosts'  # Label for sidebar
        
        # Preprocess same as main dataset
        df_full_original = final_preprocess_and_map_columns(original_posts_df)
        df_full_original['timestamp_share'] = df_full_original['timestamp_share'].apply(parse_timestamp_robust)
    else:
        # Fallback: empty dataframe with same columns as main dataset
        df_full_original = pd.DataFrame(columns=['account_id','content_id','object_id','URL','timestamp_share','Platform','original_text','Outlet','Channel','cluster','source_dataset','Sentiment'])
        
    st.sidebar.markdown("### Data Sources (Raw Count)")
    source_counts = combined_raw['source_dataset'].value_counts()
    st.sidebar.dataframe(
        source_counts.reset_index().rename(columns={'index':'Source', 'source_dataset':'Posts'}), 
        width='stretch',  # ✅ Updated from use_container_width
        hide_index=True
    )
    # Preprocess
    df_full = final_preprocess_and_map_columns(combined_raw)
    df_full['timestamp_share'] = df_full['timestamp_share'].apply(parse_timestamp_robust)

    openmask = df_full['source_dataset'] == 'OpenMeasure'
    if openmask.any():
        # Force parse OpenMeasure timestamps with more flexible formats
        df_full.loc[openmask, 'timestamp_share'] = pd.to_datetime(
            df_full.loc[openmask, 'timestamp_share'], 
            errors='coerce', 
            infer_datetime_format=True, 
            utc=True
        )
        # Fill any remaining NaT with a fallback date if needed, or log them
        na_count = df_full[openmask & df_full['timestamp_share'].isna()].shape[0]
        if na_count > 0:
            logger.warning(f"⚠️ {na_count} OpenMeasure posts have unparseable timestamps")
            
    # Date filter
    valid_dates = df_full['timestamp_share'].dropna()
    if valid_dates.empty:
        st.error("❌ No valid timestamps found.")
        st.stop()
    
    min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    selected_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    start_date = pd.Timestamp(selected_range[0] if len(selected_range)==2 else selected_range[0], tz='UTC')
    end_date = (pd.Timestamp(selected_range[1], tz='UTC') + pd.Timedelta(days=1)) if len(selected_range)==2 else start_date + pd.Timedelta(days=1)
    
        # 1. Filter MAIN dataset (for narratives, risk, dashboard)
    filtered_df = df_full[(df_full['timestamp_share'] >= start_date) & (df_full['timestamp_share'] < end_date)].copy()

    # 2. Filter SEPARATELY LOADED Original Posts dataset (for coordination)
    if not df_full_original.empty:
        filtered_original = df_full_original[
            (df_full_original['timestamp_share'] >= start_date) & 
            (df_full_original['timestamp_share'] < end_date)
        ].copy()
    else:
        filtered_original = pd.DataFrame(columns=df_full.columns)

    # Side bar
    st.sidebar.markdown("### Platform Breakdown (Filtered Count)")
    st.sidebar.markdown(f"**Total Posts (Main Dataset):** {len(filtered_df):,}")
    st.sidebar.markdown(f"**Original Posts (Coordination Analysis):** {len(filtered_original):,}")
    platform_counts_filtered = filtered_df['Platform'].value_counts()
    st.sidebar.dataframe(
        platform_counts_filtered.reset_index().rename(columns={'index':'Platform', 'Platform':'Posts'}),
        width='stretch',  
        hide_index=True
    )
    # Clustering
    df_clustered = cached_clustering(filtered_df, eps=0.3, min_samples=2, max_features=5000) if not filtered_df.empty else pd.DataFrame()

    # LLM-Powered Narrative Summarization Pipeline
    st.sidebar.info("🤖 Generating narrative summaries via LLM...")
    all_summaries = get_ethiopia_summaries(df_clustered, filtered_df)
    if not all_summaries:
        st.sidebar.warning("⚠️ No narrative clusters generated. Check data volume or LLM API status.")
    
    # ==========================================
    #  FILTERING LOGIC (Remove noise/positive)
    # ==========================================
    noise_indicators = ["no relevant claims", "no explicit claims", "no summary generated", "llm unavailable"]
    filtered_summaries = []
    for s in all_summaries:
        context = str(s.get("Context", "")).lower()
        # Skip noise or empty summaries
        if any(ind in context for ind in noise_indicators):
            continue
        filtered_summaries.append(s)
    all_summaries = filtered_summaries


    # Metrics
    total_posts = len(filtered_df)
    unique_accounts = filtered_df['account_id'].nunique()
    top_platform = filtered_df['Platform'].mode()[0] if not filtered_df['Platform'].mode().empty else "—"
    clusters_found = len(df_clustered[df_clustered['cluster'] != -1]['cluster'].unique()) if not df_clustered.empty else 0
    last_update_time = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')
    valid_clusters_count = len(all_summaries)  
    high_virality_count = len([s for s in all_summaries if "Tier 4" in s.get("Emerging Virality","")])

        
    # === TABS ===
    tabs = st.tabs([
        "🏠 Dashboard", "📊 Insights", "🔍 Coordination", "⚠️ Risk", "📰 Narratives",
        "🕸️ Network Intelligence", "🎯 Triggers & Entities"
    ])
    
    # === TAB 1: Dashboard ===
    with tabs[0]:
        st.markdown(f"""
        This dashboard supports the early detection of information manipulation and disinformation campaigns during election periods that seek to distort public opinion by:
        1. **Detecting Emerging Narratives**
        2. **Tracking Virality**
        3. **Providing Evidence**
        
        Data is updated weekly. Last updated: **{last_update_time}**
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts Analyzed", f"{total_posts:,}") 
        col2.metric("Active Narratives", valid_clusters_count)
        col3.metric("Top Platform", top_platform)
        col4.metric("Alert Level", "🚨 High" if high_virality_count > 5 else "⚠️ Medium" if high_virality_count > 0 else "✅ Low")
        st.divider() 
    
    # === TAB 2: Insights ===
    with tabs[1]:
        st.markdown("### 🔬 Data Insights")
        # ✅ Use Ethiopia variables: filtered_df and selected_range
        st.markdown(f"**Total Rows:** `{len(filtered_df):,}` | **Date Range:** {selected_range[0]} to {selected_range[-1]}")
        
        if not filtered_df.empty:
            top_influencers = filtered_df['account_id'].value_counts().head(10)
            fig_src = px.bar(top_influencers, title="Top 10 Influencers (Total Posts)", labels={'value': 'Post Count', 'index': 'Account ID'})
            st.plotly_chart(fig_src, width='stretch', key="top_influencers")  
            
            platform_counts = filtered_df['Platform'].value_counts()
            fig_platform = px.bar(platform_counts, title="Post Distribution by Platform", labels={'value': 'Post Count', 'index': 'Platform'})
            st.plotly_chart(fig_platform, width='stretch', key="platform_dist")  
            
            social_media_df = filtered_df[~filtered_df['Platform'].isin(['Media', 'News/Media'])].copy()
            if not social_media_df.empty and 'object_id' in social_media_df.columns:
                social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
                all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
                if all_hashtags:
                    hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                    fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                    st.plotly_chart(fig_ht, width='stretch', key="top_hashtags")  
            
            plot_df = filtered_df.copy()
            plot_df = plot_df.set_index('timestamp_share')
            time_series = plot_df.resample('D').size()
            fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Total Posts', 'timestamp_share': 'Date'})
            st.plotly_chart(fig_ts, width='stretch', key="daily_volume")  
    
    # === TAB 3: Coordination ===
    with tabs[2]:
        st.markdown("### 🔍 Coordination Analysis")
        st.caption("Detecting accounts sharing **identical messages** across ≥5 unique accounts (excluding same-entity cross-posting)")
        
        if not filtered_original.empty and 'original_text' in filtered_original.columns:
            # Group by exact text content
            exact_matches = filtered_original.groupby('original_text').filter(lambda x: len(x) >= 5)
            
            if not exact_matches.empty:
                coordination_groups = []
                for text, group in exact_matches.groupby('original_text'):
                    unique_accounts = group['account_id'].dropna().unique()
                    
                    # ✅ REQUIREMENT: ≥5 unique accounts
                    if len(unique_accounts) < 5:
                        continue
                    
                    # ✅ AVOID SELF-COMPARISON
                    base_names = [re.sub(r'[^a-zA-Z0-9]', '', acc.lower())[:10] for acc in unique_accounts]
                    if len(set(base_names)) < len(unique_accounts) * 0.8:
                        continue
                    
                    coordination_groups.append({
                        'text': text,
                        'accounts': list(unique_accounts),
                        'count': len(group),
                        'platforms': group['Platform'].dropna().unique().tolist(),
                        'timestamps': group['timestamp_share'].dropna().tolist(),
                        'posts_data': group[['account_id', 'Platform', 'URL', 'timestamp_share', 'original_text']]  # ✅ Add posts data
                    })
                
                if coordination_groups:
                    st.success(f"✅ Found {len(coordination_groups)} coordination groups with identical messages across ≥5 accounts")
                    
                    for i, grp in enumerate(sorted(coordination_groups, key=lambda x: x['count'], reverse=True)[:10]):
                        with st.expander(f"🔗 Group {i+1} — {grp['count']} posts, {len(grp['accounts'])} accounts"):
                            st.code(grp['text'][:300] + "..." if len(grp['text']) > 300 else grp['text'], language=None)
                            st.markdown(f"**Accounts:** {', '.join(grp['accounts'][:10])}{'...' if len(grp['accounts']) > 10 else ''}")
                            st.markdown(f"**Platforms:** {', '.join(grp['platforms'])}")
                            if grp['timestamps']:
                                earliest = min(grp['timestamps']).strftime('%Y-%m-%d %H:%M')
                                latest = max(grp['timestamps']).strftime('%Y-%m-%d %H:%M')
                                st.caption(f"⏰ First: {earliest} → Last: {latest}")
                            
                            # ✅ SHOW ACTUAL POSTS FROM THESE ACCOUNTS
                            with st.expander("📄 View Posts from These Accounts"):
                                st.dataframe(
                                    grp['posts_data'].head(20),  # Show first 20 posts
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "URL": st.column_config.LinkColumn("Link", display_text="🔗 View"),
                                        "timestamp_share": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm")
                                    }
                                )
                else:
                    st.info("ℹ️ No coordination detected: No identical messages shared by ≥5 unique accounts (after filtering self-comparisons)")
            else:
                st.info("ℹ️ No exact duplicate messages found with ≥5 posts")
        else:
            st.info("ℹ️ No original posts data available for coordination analysis")
    
    # === TAB 4: Risk ===
    with tabs[3]:
        st.markdown("### ⚠️ Narrative Risk Overview")
        if not df_clustered.empty:
            sizes = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').size()
            if not sizes.empty:
                # Generate virality tiers safely
                virality_values = []
                for c in sizes.values:
                    try:
                        v = assign_virality_tier(int(c))
                    except:
                        v = "Tier 1: Limited"
                    virality_values.append(v if isinstance(v, str) and v.strip() else "Tier 1: Limited")
                
                risk_df = pd.DataFrame({
                    'Cluster': sizes.index, 
                    'Count': sizes.values, 
                    'Virality': virality_values
                }).dropna(subset=['Virality'])
                
                if risk_df.empty:
                    st.info("ℹ️ No valid risk data after filtering")
                else:
                    # ✅ SIMPLEST FIX: Color by numeric Count, not categorical Virality string
                    fig = px.bar(
                        risk_df.nlargest(10, 'Count'), 
                        x='Cluster', 
                        y='Count',
                        color='Count',  # ✅ Color by numeric value (avoids KeyError)
                        color_continuous_scale='Reds',  # ✅ Built-in continuous scale
                        title="Top Clusters by Volume"
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # ✅ Show virality tier in the table below
                    st.dataframe(
                        risk_df.nlargest(10, 'Count')[['Cluster', 'Count', 'Virality']], 
                        width='stretch',
                        hide_index=True
                    )
            else:
                st.info("ℹ️ No cluster data available for risk assessment")
        else:
            st.info("ℹ️ No clustering data available")      
            
    # === TAB 5: Narratives ===
    with tabs[4]:
        st.markdown("### 📰 Trending Narratives")
        
        if not all_summaries:
            st.info("ℹ️ No narrative summaries generated. Check data volume or LLM API status.")
        else:
            for summary in sorted(all_summaries, key=lambda x: x['Total_Reach'], reverse=True):
                #  Skip clusters with no meaningful content
                context = summary.get('Context', '').lower()
                if any(phrase in context for phrase in [
                    "no explicit claims", 
                    "not explicitly stated", 
                    "this account doesn't exist", 
                    "no additional information",
                    "summary generation failed",
                    "no evidence for",
                    "accounts do not exist",
                    "none"
                ]):
                    continue
                
                if all(empty_phrase in context for empty_phrase in [
                    "explicit claims: none",
                    "targeted groups: none", 
                    "language/tone: none",
                    "sample quotes: none"
                ]):
                    continue
                
                st.markdown(f"### Cluster #{summary['cluster_id']} — {summary['Emerging Virality']}")
                
                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Reach", f"{summary['Total_Reach']:,}")
                m2.metric("Amplifiers", f"{summary['Amplifiers_Count']:,}")
                m3.caption(f"Platforms: {summary['Top_Platforms']}")
                
                #  Format summary with proper line breaks
                st.markdown("**📋 Narrative Intelligence Report**")
                
                formatted_context = summary['Context']
                section_headers = [
                    'NARRATIVE THEME:', 'EXPLICIT CLAIMS:', 'TARGETED GROUPS/ENTITIES:', 
                    'LANGUAGE/TONE OBSERVED:', 'SAMPLE QUOTES:', 'Key Findings:', 
                    'Viral Slogans:', 'Foreign Interference:', 'Calls for Protests:'
                ]
                
                for header in section_headers:
                    formatted_context = formatted_context.replace(header, f"\n\n**{header}**\n")
                
                formatted_context = formatted_context.replace('- [', '- [').replace('\n-', '\n- ')
                st.markdown(formatted_context.strip())
                
                if summary['Originators'] != "Unknown":
                    st.caption(f"🔍 Originators: {summary['Originators']}")
                
                with st.expander(f"📂 View Cluster Evidence ({summary['Total_Reach']} posts)"):
                    pdf = summary['Posts_Data'].copy()
                    if 'timestamp_share' in pdf.columns:
                        pdf['Timestamp'] = pdf['timestamp_share'].dt.strftime('%Y-%m-%d %H:%M')
                    display_cols = ['Timestamp', 'Platform', 'account_id', 'object_id', 'URL'] if 'Timestamp' in pdf.columns else ['Platform', 'account_id', 'object_id', 'URL']
                    st.dataframe(
                        pdf[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={"URL": st.column_config.LinkColumn("Link", display_text="🔗 View")}
                    )
                st.divider()  
        
        # =============================================================================
        # 📱 PLATFORM-SPECIFIC TRENDING (TELEGRAM & TikTok)
        # =============================================================================
        st.divider()
        st.subheader("📱 Telegram and Tiktok Trending Content")
        st.caption("Most recent trending posts from Telegram and TikTok")
        
        #  OpenMeasure = Telegram, TikTok = TikTok
        telegram_posts = filtered_df[
            (filtered_df['Platform'] == 'Telegram') | 
            (filtered_df['source_dataset'] == 'OpenMeasure')
        ].copy()
        
        tiktok_posts = filtered_df[
            (filtered_df['Platform'] == 'TikTok') | 
            (filtered_df['source_dataset'] == 'TikTok')
        ].copy()
        
        # Create two columns for side-by-side view
        col_telegram, col_tiktok = st.columns(2)
        
        # --- TELEGRAM  SECTION ---
        with col_telegram:
            st.markdown("#### 📱 Telegram Posts ")
            
            if not telegram_posts.empty:
                telegram_posts = telegram_posts.sort_values('timestamp_share', ascending=False)
                
                for _, row in telegram_posts.head(5).iterrows():
                    date_str = row['timestamp_share'].strftime('%m/%d %H:%M') if pd.notna(row['timestamp_share']) else 'N/A'
                    account_str = str(row['account_id'])[:25]
                    
                    with st.expander(f"📄 **{account_str}** • {date_str}"):
                        st.markdown(f"**Account:** `{row['account_id']}`")
                        if pd.notna(row['timestamp_share']):
                            st.caption(f"📅 Posted: {row['timestamp_share'].strftime('%Y-%m-%d %H:%M')}")
                        
                        content = str(row['object_id']).strip()
                        if len(content) > 150:
                            st.markdown(f"**Message:** {content[:150]}...")
                            with st.expander("📖 Read full message"):
                                st.write(content)
                        else:
                            st.markdown(f"**Message:** {content}")
                        
                        if pd.notna(row['URL']) and row['URL'].startswith('http'):
                            st.markdown(f"🔗 **[Open in Telegram]({row['URL']})**")
            else:
                st.info("ℹ️ No Telegram posts (OpenMeasure) in current date range")
        
        # --- TIKTOK SECTION ---
        with col_tiktok:
            st.markdown("#### 🎵 TikTok Trending Insights")
            
            if not tiktok_posts.empty:
                # Sort by views if available, else by date
                if 'playCount' in tiktok_posts.columns:
                    tiktok_posts = tiktok_posts.sort_values('playCount', ascending=False)
                elif 'timestamp_share' in tiktok_posts.columns:
                    tiktok_posts = tiktok_posts.sort_values('timestamp_share', ascending=False)
                    
                for _, row in tiktok_posts.head(5).iterrows():
                    account = str(row.get('account_id', 'Unknown'))[:30] if pd.notna(row.get('account_id')) else 'Unknown'
                    date_str = row['timestamp_share'].strftime('%m/%d') if pd.notna(row.get('timestamp_share')) else 'N/A'
                    
                    with st.expander(f"🎵 **{account}** • {date_str}"):
                        st.markdown(f"**Account:** `{account}`")
                        
                        # ✅ Engagement Metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            if pd.notna(row.get('playCount')): st.metric("Views", f"{int(row['playCount']):,}")
                            if pd.notna(row.get('diggCount')): st.metric("Likes", f"{int(row['diggCount']):,}")
                        with col2:
                            if pd.notna(row.get('commentCount')): st.metric("Comments", f"{int(row['commentCount']):,}")
                            if pd.notna(row.get('shareCount')): st.metric("Shares", f"{int(row['shareCount']):,}")
                            
                        # ✅ Hashtags
                        tags = [row.get(f'hashtag_{i}') for i in range(5) if pd.notna(row.get(f'hashtag_{i}'))]
                        if tags:
                            st.markdown("**🔥 Hashtags:** " + " ".join([f"`#{t}`" for t in tags]))
                            
                        # ✅ Language
                        if pd.notna(row.get('textLanguage')) and str(row.get('textLanguage', '')).strip() not in ['un', 'nan', '']:
                            lang_display = {"am": "🇪🇹 Amharic", "en": "🇬🇧 English", "om": "🇪🇹 Oromo", "so": "🇸🇴 Somali"}.get(str(row['textLanguage']), str(row['textLanguage']).upper())
                            st.caption(f"🌐 Language: {lang_display}")
                            
                        # ✅ Video Link
                        if pd.notna(row.get('URL')) and str(row['URL']).startswith('http'):
                            st.markdown(f"🎬 **[Watch Video 🔗]({row['URL']})**")
                        st.divider()
            else:
                st.info("ℹ️ No TikTok posts in current date range")  
                
    # === TAB 6: Network & Coordination Intelligence ===
    with tabs[5]:
        st.subheader("🕸️ Account Coordination Network")
        st.caption("Visualizing accounts sharing identical messages. Node size = influence, edge thickness = coordination strength.")
        
        # ✅ Controls for better exploration
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            min_connections = st.slider("🔗 Minimum connections to show", min_value=1, max_value=10, value=2)
        with col2:
            top_n_nodes = st.slider("👥 Show top N accounts", min_value=10, max_value=100, value=50, step=10)
        with col3:
            layout_type = st.selectbox("🗺️ Layout style", ["spring", "circular", "kamada_kawai", "shell"], index=0)
        
        if not df_clustered.empty and 'cluster' in df_clustered.columns and not filtered_original.empty:
            # Build coordination graph from EXACT matches in original posts only
            G = nx.Graph()
            
            # Group by exact text to find coordination
            exact_matches = filtered_original.groupby('original_text').filter(lambda x: len(x) >= 2)
            
            for text, group in exact_matches.groupby('original_text'):
                accounts = group['account_id'].dropna().unique()
                if len(accounts) >= 2:
                    # Add edges between all accounts sharing this exact text
                    for i in range(len(accounts)):
                        for j in range(i+1, len(accounts)):
                            # Weight = number of shared identical messages
                            current_weight = G.get_edge_data(accounts[i], accounts[j], {}).get('weight', 0)
                            G.add_edge(accounts[i], accounts[j], weight=current_weight + 1)
            
            # Filter to nodes with minimum connections
            G = G.copy()
            nodes_to_keep = [n for n, d in G.degree() if d >= min_connections]
            G = G.subgraph(nodes_to_keep).copy()
            
            if G.number_of_edges() > 0:
                # ✅ Get top N nodes by degree (influence)
                top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n_nodes]
                top_node_names = [n for n, _ in top_nodes]
                G_top = G.subgraph(top_node_names).copy()
                
                # ✅ Choose layout algorithm
                if layout_type == "spring":
                    pos = nx.spring_layout(G_top, k=0.8, iterations=50, seed=42)
                elif layout_type == "circular":
                    pos = nx.circular_layout(G_top)
                elif layout_type == "kamada_kawai":
                    pos = nx.kamada_kawai_layout(G_top)
                else:  # shell
                    pos = nx.shell_layout(G_top)
                
                # ✅ Prepare node data with platform info and metrics
                node_data = []
                for node in G_top.nodes():
                    # Get platform from original dataset
                    platform = filtered_original[filtered_original['account_id'] == node]['Platform'].mode()
                    platform = platform.iloc[0] if not platform.empty else "Unknown"
                    
                    node_data.append({
                        'account': node,
                        'degree': G_top.degree(node),
                        'platform': platform,
                        'x': pos[node][0],
                        'y': pos[node][1]
                    })
                
                node_df = pd.DataFrame(node_data)
                
                # ✅ Color nodes by platform for easy identification
                platform_colors = {
                    'X': '#1DA1F2', 'Facebook': '#1877F2', 'TikTok': '#000000', 
                    'Telegram': '#0088cc', 'Media': '#6B7280', 'Unknown': '#9CA3AF'
                }
                node_df['color'] = node_df['platform'].map(lambda p: platform_colors.get(p, '#9CA3AF'))
                
                # ✅ Prepare edge data with thickness based on weight
                edge_data = []
                for u, v, data in G_top.edges(data=True):
                    weight = data.get('weight', 1)
                    edge_data.append({
                        'source': u, 'target': v, 'weight': weight,
                        'x0': pos[u][0], 'y0': pos[u][1],
                        'x1': pos[v][0], 'y1': pos[v][1]
                    })
                edge_df = pd.DataFrame(edge_data)
                
                # ✅ Create the visualization
                fig = go.Figure()
                
                # Add edges (thicker = stronger coordination)
                for _, edge in edge_df.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[edge['x0'], edge['x1'], None],
                        y=[edge['y0'], edge['y1'], None],
                        mode='lines',
                        line=dict(width=edge['weight'] * 1.5, color='rgba(100, 116, 139, 0.4)'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
                # Add nodes with visible labels
                fig.add_trace(go.Scatter(
                    x=node_df['x'],
                    y=node_df['y'],
                    mode='markers+text',  # ✅ Show both markers AND text labels
                    marker=dict(
                        size=node_df['degree'] * 4 + 12,  # Size by influence
                        color=node_df['color'],
                        line=dict(width=2, color='white'),
                        opacity=0.9
                    ),
                    text=node_df['account'],  # ✅ Show account names directly on nodes
                    textposition="top center",  # Position labels above nodes
                    textfont=dict(size=9, color='#1F2937', family="monospace"),  # Readable font
                    hovertemplate="<b>%{text}</b><br>" +
                                "Platform: %{customdata[0]}<br>" +
                                "Connections: %{customdata[1]}<br>" +
                                "<extra></extra>",
                    customdata=node_df[['platform', 'degree']],
                    name="Accounts"
                ))
                
                # ✅ Professional styling
                fig.update_layout(
                    title=f"Coordination Network — {G_top.number_of_nodes()} accounts, {G_top.number_of_edges()} connections",
                    height=700,  # Taller for better label visibility
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
                    margin=dict(l=40, r=40, t=80, b=40),
                    hovermode='closest',
                    showlegend=True
                )
                
                # ✅ Add legend for platform colors
                for platform, color in platform_colors.items():
                    if platform in node_df['platform'].values:
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(size=12, color=color, line=dict(width=2, color='white')),
                            name=platform,
                            showlegend=True
                        ))
                
                st.plotly_chart(fig, width='stretch')
                
                # ✅ Stats and export
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accounts Shown", G_top.number_of_nodes())
                col2.metric("Connections", G_top.number_of_edges())
                col3.metric("Avg Connections", f"{sum(dict(G_top.degree()).values())/G_top.number_of_nodes():.1f}")
                col4.metric("Strongest Link", f"{edge_df['weight'].max()} shared messages")
                
                if st.button("📥 Export Network Data"):
                    # Export in multiple formats
                    network_data = {
                        "nodes": node_df.to_dict('records'),
                        "edges": edge_df.to_dict('records'),
                        "metadata": {
                            "min_connections": min_connections,
                            "top_n_nodes": top_n_nodes,
                            "layout": layout_type,
                            "generated_at": pd.Timestamp.now().isoformat()
                        }
                    }
                    st.download_button(
                        "Download JSON",
                        json.dumps(network_data, indent=2),
                        "coordination_network.json",
                        "application/json"
                    )
                
                # ✅ Show top coordinated pairs table
                with st.expander("🔍 Top Coordinated Account Pairs"):
                    top_edges = edge_df.nlargest(10, 'weight')
                    st.dataframe(
                        top_edges[['source', 'target', 'weight']].rename(
                            columns={'source': 'Account 1', 'target': 'Account 2', 'weight': 'Shared Messages'}
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                    
            else:
                st.info(f"ℹ️ No coordination links found with ≥{min_connections} connections. Try lowering the threshold.")
        else:
            st.info("ℹ️ Upload coordination data to generate network visualization.")
        
    # === TAB 7: Triggers & Entities WITH LEXICON MANAGEMENT ===
    with tabs[6]:
        st.subheader("🎯 Trigger Terms & Targeted Entities")
        
        # ✅ SHOW TOP TRIGGER TERMS FROM DATASET (New Section)
        with st.expander("📊 Top Trigger Terms in Current Dataset", expanded=True):
            if not filtered_df.empty and 'original_text' in filtered_df.columns:
                # Scan all posts for lexicon matches
                all_matches = []
                for _, row in filtered_df.iterrows():
                    text = str(row.get('original_text', ''))
                    matches = scan_text_for_lexicon_terms(text)
                    all_matches.extend(matches)
                
                if all_matches:
                    # Aggregate by term
                    term_counts = Counter([m['term'] for m in all_matches])
                    top_terms = term_counts.most_common(10)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🔥 Most Frequent Trigger Terms**")
                        for term, count in top_terms:
                            # Find metadata for display
                            metadata = next((t for cat in CONFIG['lexicon'].values() for t in [v for k,v in cat.items() if k==term]), {})
                            severity = metadata.get('severity', 'medium')
                            severity_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(severity, "⚪")
                            st.markdown(f"{severity_icon} `{term}` — {count:,} mentions")
                    
                    with col2:
                        st.markdown("**🎯 Most Targeted Entities**")
                        entity_counts = Counter([m['target_entity'] for m in all_matches if m['target_entity']])
                        for entity, count in entity_counts.most_common(5):
                            st.markdown(f"👤 `{entity}` — {count:,} mentions")
                else:
                    st.info("ℹ️ No lexicon matches found in current date range")
            else:
                st.info("ℹ️ No data available for trigger analysis")
        
        st.divider()
        
        # 4 sub-tabs including Lexicon Management
        sub_tabs = st.tabs(["🔍 Trigger Scanner", "👥 Entity Registry", "📊 Analytics", "⚙️ Lexicon Management"])
        
        # --- Sub-tab 1: Trigger Scanner ---
        with sub_tabs[0]:
            st.markdown("### 🔍 Real-Time Trigger Detection")
            
            # Show lexicon stats
            lexicon_df = get_lexicon_as_dataframe()
            total_terms = len(lexicon_df)
            st.info(f"✅ Lexicon contains {total_terms:,} terms across {len(CONFIG['lexicon'])} categories")
            
            # Test text input with clear instructions
            test_text = st.text_area(
                "Paste text to scan for hate speech terms", 
                height=100, 
                placeholder="Example: የወያኔ ድስት ላሺ... or 'Kill them all'",
                key="trigger_scanner_input"
            )
            
            if st.button("🔎 Scan Text", key="scan_button"):
                if not test_text.strip():
                    st.warning("⚠️ Please enter text to scan")
                else:
                    with st.spinner("🔍 Scanning for trigger terms..."):
                        # Scan the text
                        matches = scan_text_for_lexicon_terms(test_text)
                        
                        if matches:
                            # Calculate risk
                            risk = calculate_risk_score(matches)
                            
                            # ✅ CLEAR FEEDBACK WITH RISK LEVEL
                            risk_color = {'low': '✅', 'medium': '⚠️', 'high': '🔴', 'critical': '🚨'}
                            risk_emoji = risk_color.get(risk['level'], '⚪')
                            
                            st.markdown(f"### {risk_emoji} Risk Assessment: {risk['level'].upper()}")
                            st.markdown(f"**Risk Score:** {risk['score']} / 15")
                            
                            # Show matched terms
                            st.markdown("**🔍 Matched Trigger Terms:**")
                            matches_df = pd.DataFrame(matches)[['term', 'category', 'severity', 'target_entity']]
                            st.dataframe(matches_df, use_container_width=True, hide_index=True)
                            
                            # Show risk breakdown
                            if risk['breakdown']:
                                st.markdown("**📊 Risk by Category:**")
                                for cat, score in risk['breakdown'].items():
                                    st.markdown(f"- {cat.replace('_', ' ').title()}: {score:.1f}")
                            
                            # Final verdict
                            if risk['level'] in ['high', 'critical']:
                                st.error(f"🚨 This text contains **{risk['term_count']} hateful/harmful trigger terms**. Review recommended.")
                            elif risk['level'] == 'medium':
                                st.warning(f"⚠️ This text contains **{risk['term_count']} potentially harmful terms**. Monitor closely.")
                            else:
                                st.success(f"✅ This text has **{risk['term_count']} low-risk terms**. Generally acceptable.")
                        else:
                            st.success("✅ No lexicon matches found — text appears clean based on current trigger list")
            
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
            
            # ✅ ADD WORDCLOUD OF TOP TRIGGER TERMS
            if not filtered_df.empty and 'original_text' in filtered_df.columns:
                with st.spinner("🎨 Generating trigger word cloud..."):
                    # Scan all posts for lexicon matches
                    all_matches = []
                    for _, row in filtered_df.iterrows():
                        text = str(row.get('original_text', ''))
                        matches = scan_text_for_lexicon_terms(text)
                        all_matches.extend(matches)
                    
                    if all_matches:
                        # Aggregate by term with frequency and severity
                        term_data = defaultdict(lambda: {'count': 0, 'severity': 'medium'})
                        for match in all_matches:
                            term = match['term']
                            term_data[term]['count'] += 1
                            # Use highest severity if term appears with multiple levels
                            current_sev = term_data[term]['severity']
                            new_sev = match['severity']
                            severity_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                            if severity_order.get(new_sev, 2) > severity_order.get(current_sev, 2):
                                term_data[term]['severity'] = new_sev
                        
                        # Prepare data for wordcloud (frequency-based sizing)
                        word_freq = {term: data['count'] for term, data in term_data.items()}
                        
                        # Generate and display wordcloud
                        wordcloud = generate_trigger_wordcloud(
                            {'top_terms': [{'term': t, 'count': c} for t, c in word_freq.items()]},
                            width=800, height=400
                        )
                        if wordcloud:
                            img_base64 = wordcloud_to_base64(wordcloud)
                            st.image(f"data:image/png;base64,{img_base64}", caption="🔥 Top Trigger Terms by Frequency", use_container_width=True)
                        
                        # Show top terms table below wordcloud
                        top_terms = sorted(term_data.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
                        st.markdown("**Top 15 Trigger Terms**")
                        term_df = pd.DataFrame([
                            {'Term': term, 'Mentions': data['count'], 'Severity': data['severity'].title()}
                            for term, data in top_terms
                        ])
                        st.dataframe(term_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("ℹ️ No lexicon matches found in current date range")
            else:
                st.info("ℹ️ No data available for trigger analysis")
            
            st.divider()
        
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
    
   
if __name__ == "__main__":
    main()
