import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import time
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from core.config import DATA_CSV
from core.data_prep import load_inventory_df, to_rag_sentences
from core.embedding_store import build_or_load_tfidf
from core.rag_chain import make_rag
from core.ml_model import train_or_load, predict, basic_stats

# -------------- Setup --------------
load_dotenv()
st.set_page_config(page_title="Inventory RAG Assistant", layout="wide", page_icon="🧠")

# ---- Custom CSS for a sleek look ----
st.markdown("""
<style>
/* Global tweaks */
:root {
  --card-bg: rgba(255, 255, 255, 0.65);
  --card-border: rgba(0,0,0,.05);
}
[data-testid="stSidebar"] .stButton>button {
  width: 100%;
  border-radius: 12px;
}
.stMetric {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 12px;
}
.block-container {
  padding-top: 1.6rem;
}

/* Card-like containers */
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 18px 18px 8px 18px;
  margin-bottom: 12px;
}

/* Source chip */
.source-chip {
  display:inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid var(--card-border);
  margin-right: 6px;
  margin-bottom: 6px;
  font-size: 12px;
  background: rgba(127,127,127,.08);
}

/* Answer content */
.answer {
  font-size: 1.02rem;
  line-height: 1.6;
}

/* Section headers */
h2, h3 {
  margin-top: 0.2rem;
}

/* File uploader note */
.uploader-note {
  font-size: 12px;
  opacity: 0.8;
  margin-top: -6px;
}

/* Buttons */
button[kind="primary"] {
  border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------- State --------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user/assistant","content":"...","sources":[...] }]

# -------------- Helpers --------------
@st.cache_data(show_spinner=False)
def _load_df_cached(path: str):
    df = load_inventory_df(path)
    return df

def _rebuild_tfidf_and_retrain(df: pd.DataFrame):
    docs = to_rag_sentences(df)
    build_or_load_tfidf(docs)      # builds first time or loads existing
    train_or_load()                # trains model if not cached yet

def _prepare_app_start():
    with st.spinner("Loading data • building TF-IDF • warming up model..."):
        df = _load_df_cached(DATA_CSV)
        _rebuild_tfidf_and_retrain(df)
    return df

def _kpi(metric_label: str, value, delta=None, help_text=None, col=None):
    if col is None:
        st.metric(metric_label, value, delta=delta, help=help_text)
    else:
        with col:
            st.metric(metric_label, value, delta=delta, help=help_text)

def _download_df(name: str, df: pd.DataFrame):
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    return st.download_button(
        label=f"⬇️ Download {name}.csv",
        data=bio.getvalue(),
        file_name=f"{name}.csv",
        mime="text/csv",
        use_container_width=True
    )

# -------------- Header --------------
title_left, title_right = st.columns([0.7, 0.3])

with title_left:
    st.markdown("### 🧠 RAG Enhanced AI framework  for smart inventory in rice retail")
    st.caption("RAG over your inventory + analytics + ML demand prediction.")
with title_right:
    st.markdown("""
    <div class="card">
      <div><b>Status</b></div>
      <div style="font-size:13px;opacity:.9;">Loaded: TF-IDF index & ML model</div>
      <div style="font-size:13px;opacity:.9;">LLM: Groq via LangChain</div>
    </div>
    """, unsafe_allow_html=True)

# -------------- Sidebar --------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.write("Use these tools to manage your data and model.")
    k = st.slider("Retrieval results (k)", min_value=1, max_value=15, value=5, step=1)

    uploaded = st.file_uploader("Upload CSV to replace dataset", type=["csv"])
    st.caption('<div class="uploader-note">The file will overwrite data/high_demand_inventory_dataset.csv</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        rebuild = st.button("🔁 Rebuild TF-IDF", use_container_width=True)
    with col_b:
        retrain = st.button("🧪 Retrain Model", use_container_width=True)

# -------------- Initialize data/models --------------
df = _prepare_app_start()

# -------------- CSV Upload handling --------------
if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        # Save to disk path used by the app
        os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
        new_df.to_csv(DATA_CSV, index=False)
        # Clear cache & rebuild
        _load_df_cached.clear()
        df = _load_df_cached(DATA_CSV)
        _rebuild_tfidf_and_retrain(df)
        st.success("✅ Dataset replaced, TF-IDF rebuilt, and model retrained.")
    except Exception as e:
        st.error(f"Upload failed: {e}")

# -------------- Rebuild / Retrain actions --------------
if rebuild:
    try:
        docs = to_rag_sentences(df)
        build_or_load_tfidf(docs)
        st.success("✅ TF-IDF index rebuilt.")
    except Exception as e:
        st.error(f"Rebuild failed: {e}")

if retrain:
    try:
        # Remove cached model to force fresh train (optional; train_or_load trains if not present)
        # import pathlib; p = pathlib.Path("models/demand_model.pkl"); 
        # if p.exists(): p.unlink()
        train_or_load()
        st.success("✅ Model retrained and cached.")
    except Exception as e:
        st.error(f"Retrain failed: {e}")

# -------------- Tabs --------------
tab_chat, tab_analytics, tab_predict, tab_data = st.tabs(
    ["💬 Chat Assistant", "📈 Analytics", "🤖 Predictor", "🗃️ Data Explorer"]
)

# ==========================
# 💬 Chat Assistant
# ==========================
with tab_chat:
    st.markdown("#### Ask questions about your inventory")
    st.caption("RAG uses TF-IDF on compact row summaries. The LLM answers only from retrieved context.")
    # Chat history UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="answer">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        meta = s.get("metadata", {})
                        chip = f"<span class='source-chip'>{meta.get('product_name','?')} | {meta.get('brand','?')} | {meta.get('date','')}</span>"
                        st.markdown(chip, unsafe_allow_html=True)
                    st.write("---")
                    for s in msg["sources"]:
                        st.caption(s.get("text",""))

    # Chat input
    prompt = st.chat_input("Type your question (e.g., 'Which brand has the highest average demand score?')")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag = make_rag(k=k)
                    out = rag(prompt)
                    answer = out.get("answer", "")
                    sources = out.get("sources", [])
                    st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)

                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                meta = s.get("metadata", {})
                                chip = f"<span class='source-chip'>{meta.get('product_name','?')} | {meta.get('brand','?')} | {meta.get('date','')}</span>"
                                st.markdown(chip, unsafe_allow_html=True)
                            st.write("---")
                            for s in sources:
                                st.caption(s.get("text",""))

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(str(e))

# ==========================
# 📈 Analytics
# ==========================
with tab_analytics:
    st.markdown("#### Inventory Overview")
    try:
        stats = basic_stats()
        col1, col2, col3, col4 = st.columns(4)
        _kpi("Rows", f"{stats['rows']:,}", col=col1)
        _kpi("Brands", str(len(stats["avg_demand_by_brand"])), col=col2)
        _kpi("Categories", str(len(stats["avg_demand_by_category"])), col=col3)
        _kpi("Top Restocks (shown)", "10", col=col4)

        st.write("")
        grid1 = st.columns(2)
        with grid1[0]:
            st.markdown("##### Avg demand by brand")
            df_b = pd.DataFrame(list(stats["avg_demand_by_brand"].items()), columns=["brand","avg_demand"])
            fig_b = px.bar(df_b, x="brand", y="avg_demand", title=None)
            st.plotly_chart(fig_b, use_container_width=True)

        with grid1[1]:
            st.markdown("##### Avg demand by category")
            df_c = pd.DataFrame(list(stats["avg_demand_by_category"].items()), columns=["category","avg_demand"])
            fig_c = px.bar(df_c, x="category", y="avg_demand", title=None)
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("##### Most Restocked Products")
        df_r = pd.DataFrame(stats["most_restocked"])
        # Optional filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            f_brand = st.selectbox("Filter by brand", ["All"] + sorted(df_r["brand"].unique().tolist()))
        with fcol2:
            f_min_times = st.slider("Min times restocked", min_value=0, max_value=int(df_r["times_restocked"].max() or 0), value=0)
        df_r_filtered = df_r.copy()
        if f_brand != "All":
            df_r_filtered = df_r_filtered[df_r_filtered["brand"] == f_brand]
        df_r_filtered = df_r_filtered[df_r_filtered["times_restocked"] >= f_min_times]
        st.dataframe(df_r_filtered, use_container_width=True, height=320)

    except Exception as e:
        st.error(str(e))

# ==========================
# 🤖 Demand Predictor
# ==========================
with tab_predict:
    st.markdown("#### Predict Demand Score")
    st.caption("Fill the fields. The model was trained on your CSV and includes Total Volume Moved as a feature.")

    with st.form("pred_form", clear_on_submit=False):
        left, right = st.columns(2)

        with left:
            cats = sorted(df["category"].dropna().unique().tolist()) or ["Other"]
            category = st.selectbox("Category", cats, key="pred_category")
            brands = sorted(df["brand"].dropna().unique().tolist()) or ["Generic"]
            brand = st.selectbox("Brand", brands, key="pred_brand")

            def default(col, fallback):
                try:
                    return float(df[col].median())
                except Exception:
                    return fallback

            initial_stock = st.number_input("Initial Stock", min_value=0.0, value=default("initial_stock", 100.0), step=10.0, key="pred_initial_stock")
            quantity_sold = st.number_input("Quantity Sold", min_value=0.0, value=default("quantity_sold", 60.0), step=5.0, key="pred_quantity_sold")

        with right:
            times_restocked = st.number_input("Times Restocked", min_value=0.0, value=default("times_restocked", 4.0), step=1.0, key="pred_times_restocked")
            quantity_restocked = st.number_input("Quantity Restocked", min_value=0.0, value=default("quantity_restocked", 200.0), step=10.0, key="pred_quantity_restocked")
            total_volume_moved = st.number_input("Total Volume Moved", min_value=0.0, value=default("total_volume_moved", 260.0), step=10.0, key="pred_total_volume_moved")

        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        payload = {
            "category": category,
            "brand": brand,
            "initial_stock": initial_stock,
            "quantity_sold": quantity_sold,
            "times_restocked": times_restocked,
            "quantity_restocked": quantity_restocked,
            "total_volume_moved": total_volume_moved
        }
        with st.spinner("Scoring..."):
            try:
                yhat = predict(payload)
                # Result card
                c1, c2, c3 = st.columns([0.4, 0.3, 0.3])
                with c1:
                    st.markdown("<div class='card'><b>Predicted Demand Score</b><br><span style='font-size:28px;'>"
                                f"{yhat:.2f}</span></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("<div class='card'><b>Inputs</b>", unsafe_allow_html=True)
                    st.json(payload)
                    st.markdown("</div>", unsafe_allow_html=True)
                with c3:
                    st.markdown("<div class='card'><b>Tip</b><br><span style='font-size:13px;opacity:.9;'>"
                                "Try tweaking restocks/volume to see sensitivity.</span></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

# ==========================
# 🗃️ Data Explorer
# ==========================
with tab_data:
    st.markdown("#### Explore your dataset")
    st.caption("Filter, preview, and download your inventory data.")
    # Light filters
    df_view = df.copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        brand_sel = st.selectbox("Brand", ["All"] + sorted(df_view["brand"].dropna().unique()))
    with c2:
        cat_sel = st.selectbox("Category", ["All"] + sorted(df_view["category"].dropna().unique()))
    with c3:
        min_demand = st.slider("Min Demand Score", min_value=int(df_view["demand_score"].min() or 0),
                               max_value=int(df_view["demand_score"].max() or 100), value=0)

    if brand_sel != "All":
        df_view = df_view[df_view["brand"] == brand_sel]
    if cat_sel != "All":
        df_view = df_view[df_view["category"] == cat_sel]
    df_view = df_view[df_view["demand_score"] >= min_demand]

    st.dataframe(df_view, use_container_width=True, height=380)
    _download_df("inventory_filtered", df_view)

# -------------- Footer --------------
st.write("")
st.caption("© Inventory RAG Assistant")
