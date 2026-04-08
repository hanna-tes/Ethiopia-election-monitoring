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
import textwrap
from collections import defaultdict, Counter
import json

# --- Clear Streamlit Cache on Startup ---
def clear_streamlit_cache():
    cache_dir = ".streamlit/cache"
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            st.info("✅ Streamlit cache cleared. Running fresh code.")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

clear_streamlit_cache()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config ---
CONFIG = {
    "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
    "bertrend": {"min_cluster_size": 3},
    "analysis": {"time_window": "48H"},
    "coordination_detection": {"threshold": 0.85, "max_features": 5000},
    # NEW: Trigger lexicon & PEP entity config
    "triggers": {
        "election_terms": ["election", "vote", "ballot", "poll", "candidate", "incumbent", "opposition", "rigging", "fraud", "IEC", "electoral"],
        "violence_terms": ["protest", "clash", "arrest", "detain", "violence", "unrest", "security", "military", "police"],
        "foreign_interference": ["foreign", "international", "EU", "US", "China", "Russia", "donor", "aid", "influence"],
        "economic_distress": ["economy", "inflation", "unemployment", "poverty", "debt", "crisis", "austerity"]
    },
    "pep_patterns": {
        "titles": [r'\b(President|Minister|Senator|MP|Governor|Mayor|Chief|Director|Secretary)\b', 
                   r'\b(Hon\.|Dr\.|Prof\.|Ambassador)\s+[A-Z][a-z]+'],
        "names": [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', r'\b[A-Z]{2,}\s+[A-Z][a-z]+\b']  # Simple name patterns
    }
}

# --- Groq Setup ---
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    if GROQ_API_KEY:
        import groq
        client = groq.Groq(api_key=GROQ_API_KEY)
    else:
        logger.warning("GROQ_API_KEY not found in st.secrets. LLM functions disabled.")
        client = None
except Exception as e:
    logger.warning(f"Groq client setup failed: {e}")
    client = None

# --- URLs ---
CFA_LOGO_URL = "https://opportunities.codeforafrica.org/wp-content/uploads/sites/5/2015/11/1-Zq7KnTAeKjBf6eENRsacSQ.png"
MELTWATER_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/Co%CC%82te_dIvoire_OR_Ivory_Coast_OR_Abidjan_OR_Ivoirien%20-%20Nov%205%2C%202025%20-%203%2058%2048%20PM.csv"
CIVICSIGNALS_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/cote-d-ivoire-or-ivory-all-story-urls-20251105131200.csv"
TIKTOK_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/TikTok_Oct_Nov.csv"
OPENMEASURES_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/open-measures-data%20(4).csv"
ORIGINAL_POSTS_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/Co%CC%82te_dIvoire_OR_Ivory_Coast_OR_Abidjan_OR_Ivoirien%20-%20Jan%2029%2C%202026%20-%205%2021%2000%20PM.csv"

# --- Helper Functions ---
def load_data_robustly(url, name, default_sep=','):
    df = pd.DataFrame()
    if not url:
        return df
    attempts = [
        (',', 'utf-8'),
        (',', 'utf-8-sig'),
        ('\t', 'utf-8'),
        (';', 'utf-8'),
        ('\t', 'utf-16'),
        (',', 'latin-1'),
    ]
    for sep, enc in attempts:
        try:
            df = pd.read_csv(url, sep=sep, low_memory=False, on_bad_lines='skip', encoding=enc)
            if not df.empty and len(df.columns) > 1:
                logger.info(f"✅ {name} loaded successfully (Sep: '{sep}', Enc: '{enc}', Shape: {df.shape})")
                return df
        except Exception:
            pass
    logger.error(f"❌ {name} failed to load with all combinations.")
    return pd.DataFrame()

def safe_llm_call(prompt, max_tokens=2048):
    if client is None:
        logger.warning("Groq client not initialized. LLM call skipped.")
        return None
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        try:
            content = response.choices[0].message.content.strip()
        except (AttributeError, KeyError, TypeError):
            content = str(response)
        return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

def translate_text(text, target_lang="en"):
    if client is None:
        return text
    try:
        prompt = f"Translate the following text to {target_lang}:\n{text}"
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=512
        )
        if response and hasattr(response, 'choices'):
            return response.choices[0].message.content.strip()
        return text
    except Exception:
        return text

def infer_platform_from_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    
    url = url.lower()
    
    platforms = {
        "tiktok.com": "TikTok", "vt.tiktok.com": "TikTok",
        "facebook.com": "Facebook", "fb.watch": "Facebook", "fb.com": "Facebook",
        "twitter.com": "X", "x.com": "X",
        "youtube.com": "YouTube", "youtu.be": "YouTube",
        "instagram.com": "Instagram", "instagr.am": "Instagram",
        "telegram.me": "Telegram", "t.me": "Telegram", "telegram.org": "Telegram"
    }
    
    for key, val in platforms.items():
        if key in url:
            return val
            
    media_domains = [
        "nytimes.com", "bbc.com", "cnn.com", "reuters.com", 
        "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"
    ]
    if any(domain in url for domain in media_domains):
        return "News/Media"
        
    return "Media"

def extract_original_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    cleaned = re.sub(r'^(RT|rt|QT|qt|repost|shared|via|credit)\s*[:@]\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{4}\b', '', cleaned)
    cleaned = re.sub(r"[\n\r\t]", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned.lower()

def is_original_post(text):
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    lower_text = text.strip().lower()
    if not lower_text:
        return False

    exclude_patterns = [
        r'^🔁.*reposted',
        r'\b(reposted|reshared|retweeted)\b',
        r'^(rt|qt|repost|shared|forwarded)\s*[:@\s]',
        r'^\s*([🔁↪️➡️🔄♻️]|rt|qt|repost|shared)\s*@?\w*',
        r'(\b|_)(shared|forwarded|credit|via)\s+(by\s+)?@?\w*', 
        r'(\b|_)(by|cc)\s+@',
        r'\b(?:reposted|reshared|retweeted)\b',
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, lower_text, flags=re.IGNORECASE):
            return False

    text_without_urls_mentions = re.sub(r'http\S+|\@\w+', '', text).strip()
    if len(text_without_urls_mentions) < 15:
        return False

    if len(lower_text) < 20:
        return False

    if re.search(r'^\s*("|\u201c)|"\s*@', lower_text, flags=re.IGNORECASE):
        return False

    if re.search(r'(^|\n)\s*@\w+\s*[":]', lower_text, flags=re.IGNORECASE):
        return False

    return True
    
def is_definitely_retweet(text):
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text_lower = text.lower().strip()
    
    retweet_patterns = [
        r'^\s*(rt|qt|repost|reshare|via|shared|forwarded)\s*[@:]',
        r'(@\w+\s*)+\s*(said|writes|states|commented):?',
        r'(@\w+\s*)+\s*[":]',
        r'(quoted|quoting|reposted|retweeted|shared|via|via\s+@)',
        r'(\s|^)(转发|转推|repost|partager)(\s|$)',
        r'^\s*"\s*@',
        r'^\s*@.*?"',
    ]
    
    for pattern in retweet_patterns:
        if re.search(pattern, text_lower):
            return True
    
    content_without_urls_mentions = re.sub(r'http\S+|\@\w+|#\w+', '', text).strip()
    if len(content_without_urls_mentions) < 20:
        return True
    
    return False

def is_truly_original_post(row):
    post_type = str(row.get('type', '')).lower()
    if post_type in ['retweet', 'repost', 'share', 'quoted_status']:
        return False
        
    if pd.notna(row.get('retweeted_status_id')) or pd.notna(row.get('parent_id')):
        return False
        
    if 'author_handle' in row and 'account_handle' in row:
        if row['author_handle'] != row['account_handle']:
            return False
            
    return True

def parse_timestamp_robust(timestamp):
    if pd.isna(timestamp):
        return pd.NaT
    ts_str = str(timestamp).strip()
    ts_str = re.sub(r'\s+GMT$', '', ts_str, flags=re.IGNORECASE)
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True)
        if pd.notna(parsed):
            return parsed
    except Exception:
        pass
    date_formats = [
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
        '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M',
        '%b %d, %Y %H:%M', '%d %b %Y %H:%M',
        '%A, %d %b %Y %H:%M:%S',
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'
    ]
    for fmt in date_formats:
        try:
            dayfirst = '%d' in fmt and ('%m' in fmt) and (fmt.startswith('%d'))
            parsed = pd.to_datetime(ts_str, format=fmt, errors='coerce', utc=True, dayfirst=dayfirst)
            if pd.notna(parsed):
                return parsed
        except Exception:
            continue
    return pd.NaT

# --- Combine Datasets ---
def combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df=None, openmeasures_df=None):
    combined_dfs = []
    def get_col(df, cols):
        df_cols = [c.lower().strip() for c in df.columns]
        for col in cols:
            normalized_col = col.lower().strip()
            if normalized_col in df_cols:
                return df[df.columns[df_cols.index(normalized_col)]]
        return pd.Series([np.nan]*len(df), index=df.index)
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id', 'id'])
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'headline', 'article body', 'text', 'content']) 
        mw['URL'] = get_col(meltwater_df, ['url'])
        mw_primary_dt = get_col(meltwater_df, ['date'])
        mw_alt_date = get_col(meltwater_df, ['alternate date format'])
        mw_time = get_col(meltwater_df, ['time'])
        if not mw_primary_dt.empty and len(mw_primary_dt)==len(meltwater_df):
            mw['timestamp_share'] = mw_primary_dt
        elif not mw_alt_date.empty and not mw_time.empty and len(mw_alt_date)==len(meltwater_df):
            mw['timestamp_share'] = mw_alt_date.astype(str)+' '+mw_time.astype(str)
        else:
            mw['timestamp_share'] = mw_alt_date
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'author', 'username', 'user'])
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'post_id', 'id', 'content_id'])
        cs['object_id'] = get_col(civicsignals_df, ['title', 'text', 'content', 'body', 'message', 'description', 'caption'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link', 'post_url'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'timestamp', 'date', 'created_at', 'post_date'])
        cs['source_dataset'] = 'Civicsignal'
        combined_dfs.append(cs)
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['object_id'] = get_col(tiktok_df, ['text', 'Transcript', 'caption', 'description', 'content'])
        tt['account_id'] = get_col(tiktok_df, ['authorMeta.name', 'username', 'creator', 'author'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'post_id', 'itemId'])
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'TikTok Link', 'link', 'video_url', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date', 'created_time', 'createTime'])
        tt['source_dataset'] = 'TikTok'
        combined_dfs.append(tt)
    if openmeasures_df is not None and not openmeasures_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasures_df, ['context_name'])
        om['content_id'] = get_col(openmeasures_df, ['id'])
        om['object_id'] = get_col(openmeasures_df, ['text'])
        om['URL'] = get_col(openmeasures_df, ['url'])
        om['timestamp_share'] = get_col(openmeasures_df, ['created_at'])
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)
    if not combined_dfs:
        return pd.DataFrame()
    return pd.concat(combined_dfs, ignore_index=True)

def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    if df.empty:
        return pd.DataFrame(columns=[
            'account_id', 'content_id', 'object_id', 'URL', 'timestamp_share',
            'Platform', 'original_text', 'Outlet', 'Channel', 'cluster',
            'source_dataset', 'Sentiment'
        ])
    
    df_processed = df.copy()
    if 'Sentiment' in df_processed.columns:
        df_processed = df_processed[df_processed['Sentiment'].isin(['Negative', 'Neutral'])]
        
    if 'object_id' in df_processed.columns:
        mask = df_processed['object_id'].apply(is_original_post) & \
               (~df_processed['object_id'].str.contains('🔁', na=False)) & \
               (~df_processed['object_id'].str.startswith('RT @', na=False))
        df_processed = df_processed[mask].copy()
    
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan','').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""]
    
    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    else:
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan','').fillna('')
        
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    
    if 'source_dataset' in df_processed.columns:
        tiktok_patterns = ['TikTok', 'tiktok', 'vt.tiktok', 'tiktok.com']
        for pattern in tiktok_patterns:
            df_processed.loc[df_processed['source_dataset'].str.contains(pattern, case=False, na=False), 'Platform'] = 'TikTok'
        
        telegram_patterns = ['Telegram', 'telegram', 't.me', 'telegram.org']
        for pattern in telegram_patterns:
            df_processed.loc[df_processed['source_dataset'].str.contains(pattern, case=False, na=False), 'Platform'] = 'Telegram'
        
        media_patterns = ['Media', 'News', 'Civicsignal', 'News/Media']
        for pattern in media_patterns:
            df_processed.loc[df_processed['source_dataset'].str.contains(pattern, case=False, na=False), 'Platform'] = 'Media'
    
    df_processed['Outlet'] = np.nan
    df_processed['Channel'] = np.nan
    df_processed['cluster'] = -1
    
    if 'Sentiment' not in df_processed.columns:
        df_processed['Sentiment'] = np.nan
        
    columns_to_keep = [
        'account_id', 'content_id', 'object_id', 'URL', 'timestamp_share',
        'Platform', 'original_text', 'Outlet', 'Channel', 'cluster',
        'source_dataset', 'Sentiment'
    ]
    
    final_cols = [c for c in columns_to_keep if c in df_processed.columns]
    return df_processed[final_cols].copy()

@st.cache_data(show_spinner=False)
def cached_clustering(df, eps, min_samples, max_features):
    if df.empty or 'original_text' not in df.columns:
        return pd.DataFrame()

    df_filtered = df[df['original_text'].str.len() > 15].copy()
    
    if df_filtered.empty:
        return df
    
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(3,5), 
        max_features=max_features
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(df_filtered['original_text'])
    except ValueError:
        return df
        
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    df_filtered['cluster'] = clustering.fit_predict(tfidf_matrix)
    
    df_out = df.copy()
    df_out['cluster'] = -1
    df_out.loc[df_filtered.index, 'cluster'] = df_filtered['cluster']
    
    return df_out

def assign_virality_tier(post_count):
    if post_count>=500:
        return "Tier 4: Viral Emergency"
    elif post_count>=100:
        return "Tier 3: High Spread"
    elif post_count>=20:
        return "Tier 2: Moderate"
    else:
        return "Tier 1: Limited"

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def summarize_cluster(texts, urls, cluster_data, min_ts, max_ts):
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""
    prompt = f"""
Generate a structured IMI intelligence report on online narratives related to election in Côte d'Ivoire.
Focus on pre and post election tensions and emerging narratives, including:
- Allegations of political suppression
- Electoral Commission corruption or bias
- Economic distress or state fund misuse
- Hate speech, tribalism, xenophobia
- Gender-based attacks
- Foreign interference ("Western puppet", anti-EU, etc.)
- Marginalization of minorities
- Claims of election fraud, rigging, tally center issues
- Calls for protests or civic resistance
- Viral slogans or hashtags
**Strict Instructions:**
- Only report claims **explicitly present** in the provided posts.
- Identify **originators**: accounts that first posted the core claim (from cluster_data).
- Note **amplification**: how widely it spread (Total posts).
- Do NOT invent, cut out, assume, or fact-check.
- Summarize clearly.
**Output Format (Use simple titles for normal font size):**
Narrative Title: [Short title]
Core Claim(s): [Bullet points]
Originator(s): [Account IDs or "Unknown"]
Amplification: [Total posts]
First Detected: {min_ts}
Last Updated: {max_ts}
Documents:
{joined}{url_context}
"""    
    response = safe_llm_call(prompt, max_tokens=2048)
    raw_summary = ""
    if response:
        try:
            raw_summary = response.strip()
        except Exception:
            raw_summary = str(response).strip()
            
    cleaned_summary = re.sub(r'\*\*.*?Instructions.*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL)
    cleaned_summary = re.sub(r'\*\*.*?strict.*?\*\*', '', cleaned_summary, flags=re.IGNORECASE | re.DOTALL)
    cleaned_summary = re.sub(r'```.*?```', '', cleaned_summary, flags=re.DOTALL)
    cleaned_summary = re.sub(r'###|##|#', '', cleaned_summary)
    cleaned_summary = cleaned_summary.strip()
    return cleaned_summary

def get_summaries_for_platform(df_clustered_all, filtered_df_global):
    if df_clustered_all.empty or 'cluster' not in df_clustered_all.columns:
        return []
        
    cluster_sizes = df_clustered_all[df_clustered_all['cluster'] != -1].groupby('cluster').size()
    top_15_clusters = cluster_sizes.nlargest(15).index.tolist()
    all_summaries = []
    
    for cluster_id in top_15_clusters:
        cluster_posts = df_clustered_all[df_clustered_all['cluster'] == cluster_id]
        
        all_texts = cluster_posts['object_id'].astype(str).tolist() 
        all_texts = [t for t in all_texts if len(t.strip()) > 10]
        
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
        
        raw_response = summarize_cluster(
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
            "Context":
