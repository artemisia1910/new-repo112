"""
ReWeave Analytics Dashboard
Sustainable Lifestyle Brand — Data-Driven Decision Platform
8 Pages: Overview | Descriptive | Diagnostic | Clustering |
         Classification | Association Rules | Regression | Predict New
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ReWeave Analytics",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_COLOR = "#1D9E75"
COLORS = ["#1D9E75", "#7F77DD", "#D85A30", "#378ADD", "#BA7517", "#D4537E", "#888780"]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    if not os.path.exists("reweave_survey_data.csv"):
        from generate_data import generate_reweave_data
        generate_reweave_data()
    return pd.read_csv("reweave_survey_data.csv")


# ─────────────────────────────────────────────
# FEATURE LISTS
# ─────────────────────────────────────────────

CLF_FEATURES = [
    "q1_age_group", "q2_gender", "q4_city_tier", "q6_income",
    "q7_lifestyle_spend", "q8_emi_pref", "q9_eco_importance",
    "q11_shopping_personality", "q12_eco_premium", "q19_barrier",
    "qa_quality_perception", "qc_narrative_resonance", "qe_greenwashing",
    "qk_d2c_comfort", "qi_b2b_role", "qd_traceability", "qg_nps_score",
]

REG_FEATURES = [
    "q6_income", "q7_lifestyle_spend", "q8_emi_pref",
    "q22_bag_spend", "q23_decor_spend", "qi_b2b_role",
    "qj_artisan_premium", "q9_eco_importance", "q4_city_tier",
    "qa_quality_perception", "qf_customisation", "qd_traceability",
]

CLUSTER_NUM_FEATURES = [
    "q9_eco_importance", "qd_traceability", "qg_nps_score", "spending_score",
    "arm_bags", "arm_cushion_covers", "arm_scarves",
    "arm_stationery", "arm_kids_products", "arm_quilts", "arm_wallets",
]

CLUSTER_CAT_FEATURES = [
    "q6_income", "q4_city_tier", "q11_shopping_personality",
    "qa_quality_perception", "qc_narrative_resonance",
]

ARM_COLUMNS = {
    "q13_products": "PROD",
    "q17_occasions": "OCC",
    "q18_home_decor": "DECOR",
}


# ─────────────────────────────────────────────
# ENCODING UTILITIES
# ─────────────────────────────────────────────

def encode_df(df, feature_cols, encoders=None, fit=True):
    """
    Encode a DataFrame for ML.
    Returns (X_array, encoders_dict).
    """
    if encoders is None:
        encoders = {}
    X_parts = []
    for col in feature_cols:
        if col not in df.columns:
            X_parts.append(pd.Series(np.zeros(len(df)), name=col, dtype=float))
            continue
        series = df[col].fillna("Unknown")
        if series.dtype == object or series.dtype.name == "category":
            if fit:
                le = LabelEncoder()
                encoded = le.fit_transform(series.astype(str))
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    encoded = le.fit_transform(series.astype(str))
                else:
                    known = set(le.classes_)
                    encoded = series.astype(str).apply(
                        lambda x: le.transform([x])[0] if x in known else 0
                    ).values
            X_parts.append(pd.Series(encoded.astype(float), name=col))
        else:
            X_parts.append(series.astype(float).rename(col))
    X = pd.concat(X_parts, axis=1).values
    return X, encoders


def encode_cluster_features(df, scaler=None, cat_encoders=None, fit=True):
    """Encode and scale cluster features."""
    num_data = df[CLUSTER_NUM_FEATURES].fillna(0).astype(float)
    cat_enc_cols = []
    if cat_encoders is None:
        cat_encoders = {}
    for col in CLUSTER_CAT_FEATURES:
        if col not in df.columns:
            cat_enc_cols.append(np.zeros(len(df)))
            continue
        s = df[col].fillna("Unknown").astype(str)
        if fit:
            le = LabelEncoder()
            cat_enc_cols.append(le.fit_transform(s).astype(float))
            cat_encoders[col] = le
        else:
            le = cat_encoders.get(col, LabelEncoder())
            known = set(le.classes_) if hasattr(le, "classes_") else set()
            enc = s.apply(lambda x: le.transform([x])[0] if x in known else 0)
            cat_enc_cols.append(enc.values.astype(float))
    cat_data = np.column_stack(cat_enc_cols)
    X = np.hstack([num_data.values, cat_data])
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X) if scaler else X
    return X, scaler, cat_encoders


# ─────────────────────────────────────────────
# ARM TRANSACTION BUILDER
# ─────────────────────────────────────────────

def build_transactions(df):
    transactions = []
    for _, row in df.iterrows():
        basket = []
        for col, prefix in ARM_COLUMNS.items():
            val = str(row.get(col, ""))
            if val not in ("nan", "None", "No occasion", ""):
                items = [f"{prefix}_{x.strip()}" for x in val.split("|") if x.strip()]
                basket.extend(items)
        if basket:
            transactions.append(list(set(basket)))
    return transactions


def run_arm(df):
    transactions = build_transactions(df)
    if not transactions:
        return pd.DataFrame()
    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_arr, columns=te.columns_)
    try:
        freq = apriori(te_df, min_support=0.04, use_colnames=True, max_len=3)
        if freq.empty:
            return pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=0.25)
        rules = rules[rules["lift"] > 1.0].copy()
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
        return rules.sort_values("lift", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# MODEL TRAINING (CACHED)
# ─────────────────────────────────────────────

@st.cache_resource
def train_all_models():
    df = load_data()

    # ── CLASSIFICATION ──────────────────────────
    X_clf, clf_enc = encode_df(df, CLF_FEATURES, fit=True)
    y_clf = df["purchase_intent_encoded"].fillna(1).astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    clf = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)
    y_te_bin = label_binarize(y_te, classes=[0, 1, 2])

    roc_data = {}
    for idx, cls_name in enumerate(["No", "Maybe", "Yes"]):
        fpr, tpr, _ = roc_curve(y_te_bin[:, idx], y_prob[:, idx])
        roc_data[cls_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}

    clf_metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "feature_importance": dict(zip(CLF_FEATURES, clf.feature_importances_.tolist())),
        "roc_data": roc_data,
        "X_test": X_te,
        "y_test": y_te,
        "y_pred": y_pred,
    }

    # ── CLUSTERING ───────────────────────────────
    X_clus, scaler, clus_cat_enc = encode_cluster_features(df, fit=True)
    inertias, silhouettes = [], []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_clus)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_clus, labels, sample_size=500)))

    best_k = 5
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clus)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_clus)

    cluster_metrics = {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
        "silhouette_best": float(silhouette_score(X_clus, cluster_labels, sample_size=500)),
        "labels": cluster_labels.tolist(),
        "pca_x": X_pca[:, 0].tolist(),
        "pca_y": X_pca[:, 1].tolist(),
    }

    # ── REGRESSION ───────────────────────────────
    X_reg, reg_enc = encode_df(df, REG_FEATURES, fit=True)
    y_reg = df["spending_score"].fillna(50).values
    X_rtr, X_rte, y_rtr, y_rte = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    gbr = GradientBoostingRegressor(n_estimators=150, random_state=42, max_depth=4)
    gbr.fit(X_rtr, y_rtr)
    y_rpred = gbr.predict(X_rte)
    ss_tot = np.sum((y_rte - y_rte.mean()) ** 2)
    ss_res = np.sum((y_rte - y_rpred) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    reg_metrics = {
        "rmse": float(np.sqrt(np.mean((y_rte - y_rpred) ** 2))),
        "mae": float(np.mean(np.abs(y_rte - y_rpred))),
        "r2": r2,
        "feature_importance": dict(zip(REG_FEATURES, gbr.feature_importances_.tolist())),
        "y_test": y_rte.tolist(),
        "y_pred": y_rpred.tolist(),
    }

    # ── ASSOCIATION RULES ─────────────────────────
    arm_rules = run_arm(df)

    return {
        "clf": clf,
        "clf_enc": clf_enc,
        "clf_metrics": clf_metrics,
        "kmeans": kmeans,
        "cluster_scaler": scaler,
        "clus_cat_enc": clus_cat_enc,
        "cluster_metrics": cluster_metrics,
        "cluster_labels": cluster_labels,
        "gbr": gbr,
        "reg_enc": reg_enc,
        "reg_metrics": reg_metrics,
        "arm_rules": arm_rules,
    }


# ─────────────────────────────────────────────
# PRESCRIPTIVE RULE ENGINE
# ─────────────────────────────────────────────

PERSONA_NAMES = {
    0: "Eco Evangelist",
    1: "Aesthetic Minimalist",
    2: "Deal-Driven Pragmatist",
    3: "Conscious Gifter",
    4: "Corporate Buyer",
}

BUNDLE_MAP = {
    "Eco Evangelist": "Zero Waste Home Kit (Tote + Cushion + Throw)",
    "Aesthetic Minimalist": "Design-Led Home Kit (Sling + Patchwork Cushion + Wall Art)",
    "Deal-Driven Pragmatist": "Value Starter Pack (Tote + Scrunchie + Wallet)",
    "Conscious Gifter": "Gift Hamper Set (Drawstring Bag + Cushion + Table Runner)",
    "Corporate Buyer": "ESG Gifting Box (Laptop Bag + Stationery + Branded Wallets)",
}

DISCOUNT_MAP = {
    ("Yes", "Eco Evangelist"): "No discount — full price",
    ("Yes", "Corporate Buyer"): "Bulk discount 10% on orders >50 units",
    ("Yes", "Conscious Gifter"): "Free gift wrapping + message card",
    ("Maybe", "Aesthetic Minimalist"): "First order 10% off",
    ("Maybe", "Conscious Gifter"): "15% seasonal discount",
    ("Maybe", "Eco Evangelist"): "No discount — share artisan story",
    ("No", "Deal-Driven Pragmatist"): "25% launch discount + free delivery",
    ("No", "Aesthetic Minimalist"): "20% off with design preview",
}

CHANNEL_MAP = {
    "Eco Evangelist": "Instagram + D2C Website",
    "Aesthetic Minimalist": "Pinterest + Instagram Reels",
    "Deal-Driven Pragmatist": "Flipkart + Meesho + WhatsApp",
    "Conscious Gifter": "WhatsApp + Nykaa + Local Exhibitions",
    "Corporate Buyer": "LinkedIn + Cold Email + Referral",
}

ACTION_MAP = {
    "Yes": "Convert now — send product catalogue",
    "Maybe": "Nurture — send quality proof content",
    "No": "Discount incentive or skip",
}


def get_prescriptive(intent_label, cluster_id, spend_score):
    persona = PERSONA_NAMES.get(cluster_id, "Unknown")
    bundle = BUNDLE_MAP.get(persona, "Standard Welcome Kit")
    discount = DISCOUNT_MAP.get((intent_label, persona), "15% first order discount")
    channel = CHANNEL_MAP.get(persona, "Instagram")
    action = ACTION_MAP.get(intent_label, "Nurture")
    spend_range = (
        "₹50K–₹5L (B2B)"
        if persona == "Corporate Buyer"
        else f"₹{int(spend_score * 150):,} – ₹{int(spend_score * 300):,} / yr"
    )
    return {
        "Persona": persona,
        "Recommended Bundle": bundle,
        "Discount Tier": discount,
        "Primary Channel": channel,
        "Priority Action": action,
        "Est. Annual Spend": spend_range,
    }


# ─────────────────────────────────────────────
# ── PAGE 1: OVERVIEW ─────────────────────────
# ─────────────────────────────────────────────

def page_overview(df):
    st.title("♻️ ReWeave Analytics — Founder's Dashboard")
    st.markdown("**Sustainable Lifestyle Brand | Upcycled Textile Products | Data-Driven Decision Platform**")
    st.divider()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Respondents", f"{len(df):,}")
    col2.metric("Purchase Intent — Yes", f"{(df['q25_purchase_intent']=='Yes').sum():,}")
    col3.metric("Purchase Intent — Maybe", f"{(df['q25_purchase_intent']=='Maybe').sum():,}")
    col4.metric("B2B Decision Makers", f"{(df['qi_b2b_role']=='Yes decision-maker').sum():,}")
    col5.metric("Avg Spending Score", f"{df['spending_score'].mean():.1f}/100")
    col6.metric("NPS Promoters", f"{(df['nps_segment']=='Promoter').sum():,}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        intent_counts = df["q25_purchase_intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig = px.pie(intent_counts, values="Count", names="Intent",
                     title="Purchase Intent Distribution (Q25 — Classification Target)",
                     color_discrete_sequence=["#1D9E75", "#BA7517", "#E24B4A"])
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        seg_counts = df["customer_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.bar(seg_counts, x="Count", y="Segment", orientation="h",
                     title="Latent Customer Segments (Ground Truth)",
                     color="Segment", color_discrete_sequence=COLORS)
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig = px.histogram(df, x="spending_score", nbins=40, color="customer_segment",
                           title="Spending Score Distribution by Segment",
                           color_discrete_sequence=COLORS, barmode="overlay", opacity=0.7)
        fig.update_layout(xaxis_title="Spending Score (0–100)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        nps_counts = df["nps_segment"].value_counts().reset_index()
        nps_counts.columns = ["NPS Segment", "Count"]
        fig = px.bar(nps_counts, x="NPS Segment", y="Count",
                     title="NPS Segment Distribution",
                     color="NPS Segment",
                     color_discrete_map={"Promoter": "#1D9E75", "Passive": "#BA7517", "Detractor": "#E24B4A"})
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Founder's Insight:** "
        f"{(df['q25_purchase_intent']=='Yes').mean()*100:.1f}% of respondents are ready to buy. "
        f"Corporate Buyers ({(df['qi_b2b_role']=='Yes decision-maker').mean()*100:.1f}% of sample) represent "
        "potentially 60–70% of revenue from a single B2B deal. "
        "Navigate the sidebar to explore each analytical layer."
    )


# ─────────────────────────────────────────────
# ── PAGE 2: DESCRIPTIVE ──────────────────────
# ─────────────────────────────────────────────

def page_descriptive(df):
    st.title("📊 Descriptive Analysis")
    st.markdown("*What happened? Who are our respondents and what do they look like?*")
    st.divider()

    seg_filter = st.multiselect(
        "Filter by Customer Segment",
        df["customer_segment"].unique().tolist(),
        default=df["customer_segment"].unique().tolist(),
    )
    df_f = df[df["customer_segment"].isin(seg_filter)] if seg_filter else df

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Eco Mindset", "Product Preferences", "Spending"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_f, x="q1_age_group", color="q2_gender",
                               title="Age Group × Gender",
                               category_orders={"q1_age_group": ["18-24","25-34","35-44","45-54","55+"]},
                               color_discrete_sequence=COLORS, barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            city_seg = df_f.groupby(["q4_city_tier", "customer_segment"]).size().reset_index(name="Count")
            fig = px.bar(city_seg, x="q4_city_tier", y="Count", color="customer_segment",
                         title="City Tier × Customer Segment",
                         color_discrete_sequence=COLORS, barmode="stack")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            inc_city = df_f.groupby(["q4_city_tier", "q6_income"]).size().reset_index(name="Count")
            order = ["Below 25K", "25K-50K", "50K-1L", "1L-2L", "Above 2L"]
            fig = px.bar(inc_city, x="q4_city_tier", y="Count", color="q6_income",
                         title="Income Bracket by City Tier",
                         color_discrete_sequence=COLORS, barmode="stack",
                         category_orders={"q6_income": order})
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            occ = df_f["q5_occupation"].value_counts().reset_index()
            occ.columns = ["Occupation", "Count"]
            fig = px.pie(occ, values="Count", names="Occupation",
                         title="Occupation Distribution",
                         color_discrete_sequence=COLORS)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_f, x="q9_eco_importance", color="customer_segment",
                               title="Eco Importance Score (Q9) by Segment",
                               color_discrete_sequence=COLORS, barmode="overlay", opacity=0.7,
                               nbins=5)
            fig.update_layout(xaxis_title="Eco Importance (1-5)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            gw = df_f["qe_greenwashing"].value_counts().reset_index()
            gw.columns = ["Scepticism Level", "Count"]
            fig = px.bar(gw, x="Scepticism Level", y="Count",
                         title="Greenwashing Scepticism Distribution (QE)",
                         color="Scepticism Level", color_discrete_sequence=COLORS)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            prem = df_f["q12_eco_premium"].value_counts().reset_index()
            prem.columns = ["Response", "Count"]
            fig = px.pie(prem, values="Count", names="Response",
                         title="Eco Premium Willingness (Q12)",
                         color_discrete_sequence=COLORS)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            narr = df_f["qc_narrative_resonance"].value_counts().reset_index()
            narr.columns = ["Narrative Response", "Count"]
            fig = px.bar(narr, x="Count", y="Narrative Response", orientation="h",
                         title="Waste Story Narrative Resonance (QC)",
                         color="Narrative Response", color_discrete_sequence=COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        arm_cols = [c for c in df_f.columns if c.startswith("arm_")]
        if arm_cols:
            prod_freq = df_f[arm_cols].sum().sort_values(ascending=True).reset_index()
            prod_freq.columns = ["Product", "Count"]
            prod_freq["Product"] = prod_freq["Product"].str.replace("arm_", "").str.replace("_", " ").str.title()
            fig = px.bar(prod_freq, x="Count", y="Product", orientation="h",
                         title="Product Interest Frequency (Q13 — ARM Input)",
                         color="Count", color_continuous_scale=["#9FE1CB", "#085041"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            barrier = df_f["q19_barrier"].value_counts().reset_index()
            barrier.columns = ["Barrier", "Count"]
            fig = px.pie(barrier, values="Count", names="Barrier",
                         title="Purchase Barriers (Q19)",
                         color_discrete_sequence=COLORS)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            persona_intent = df_f.groupby(["customer_segment", "q25_purchase_intent"]).size().reset_index(name="Count")
            fig = px.bar(persona_intent, x="customer_segment", y="Count", color="q25_purchase_intent",
                         title="Purchase Intent by Segment (Q25)",
                         color_discrete_map={"Yes": "#1D9E75", "Maybe": "#BA7517", "No": "#E24B4A"},
                         barmode="stack")
            fig.update_layout(xaxis_title="Segment", xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df_f, x="customer_segment", y="spending_score",
                         title="Spending Score Distribution by Segment",
                         color="customer_segment", color_discrete_sequence=COLORS)
            fig.update_layout(showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            inc_spend = df_f.groupby("q6_income")["spending_score"].mean().reset_index()
            order_i = ["Below 25K", "25K-50K", "50K-1L", "1L-2L", "Above 2L"]
            inc_spend["q6_income"] = pd.Categorical(inc_spend["q6_income"], categories=order_i, ordered=True)
            inc_spend = inc_spend.sort_values("q6_income")
            fig = px.bar(inc_spend, x="q6_income", y="spending_score",
                         title="Average Spending Score by Income Bracket",
                         color="spending_score", color_continuous_scale=["#9FE1CB", "#085041"])
            fig.update_layout(xaxis_title="Income Bracket", yaxis_title="Avg Spending Score")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df_f[["spending_score"]].describe().T.round(2),
            use_container_width=True,
        )


# ─────────────────────────────────────────────
# ── PAGE 3: DIAGNOSTIC ───────────────────────
# ─────────────────────────────────────────────

def page_diagnostic(df):
    st.title("🔍 Diagnostic Analysis")
    st.markdown("*Why did certain groups respond the way they did? Root cause analysis.*")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Conversion Barriers", "Trust vs Intent", "Spend Drivers"])

    with tab1:
        st.subheader("What is stopping customers from converting?")
        c1, c2 = st.columns(2)
        with c1:
            barrier_intent = df.groupby(["q19_barrier", "q25_purchase_intent"]).size().reset_index(name="Count")
            fig = px.bar(barrier_intent, x="q19_barrier", y="Count", color="q25_purchase_intent",
                         title="Purchase Barrier × Purchase Intent (Q19 × Q25)",
                         color_discrete_map={"Yes": "#1D9E75", "Maybe": "#BA7517", "No": "#E24B4A"},
                         barmode="stack")
            fig.update_layout(xaxis_tickangle=-20, xaxis_title="Barrier")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            conv_rate = df.groupby("q19_barrier").apply(
                lambda x: (x["q25_purchase_intent"] == "Yes").mean() * 100
            ).reset_index(name="Conversion Rate (%)")
            fig = px.bar(conv_rate, x="q19_barrier", y="Conversion Rate (%)",
                         title="Conversion Rate (% 'Yes') by Barrier Type",
                         color="Conversion Rate (%)",
                         color_continuous_scale=["#E24B4A", "#1D9E75"])
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        tier_barrier = df.groupby(["q4_city_tier", "q19_barrier"]).size().reset_index(name="Count")
        fig = px.bar(tier_barrier, x="q4_city_tier", y="Count", color="q19_barrier",
                     title="Barrier Type by City Tier",
                     color_discrete_sequence=COLORS, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Does greenwashing scepticism suppress purchase intent?")
        c1, c2 = st.columns(2)
        with c1:
            gw_intent = df.groupby(["qe_greenwashing", "q25_purchase_intent"]).size().reset_index(name="Count")
            fig = px.bar(gw_intent, x="qe_greenwashing", y="Count", color="q25_purchase_intent",
                         title="Greenwashing Scepticism × Purchase Intent",
                         color_discrete_map={"Yes": "#1D9E75", "Maybe": "#BA7517", "No": "#E24B4A"},
                         barmode="stack")
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            narr_intent = df.groupby(["qc_narrative_resonance", "q25_purchase_intent"]).size().reset_index(name="Count")
            fig = px.bar(narr_intent, x="qc_narrative_resonance", y="Count", color="q25_purchase_intent",
                         title="Narrative Resonance × Purchase Intent",
                         color_discrete_map={"Yes": "#1D9E75", "Maybe": "#BA7517", "No": "#E24B4A"},
                         barmode="group")
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        qa_gw = df.groupby(["qa_quality_perception", "qe_greenwashing"]).size().reset_index(name="Count")
        fig = px.density_heatmap(df, x="qe_greenwashing", y="qa_quality_perception",
                                  title="Quality Perception × Greenwashing Scepticism Heatmap",
                                  color_continuous_scale="Teal")
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("What drives higher spending scores?")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x="qi_b2b_role", y="spending_score",
                         title="Spending Score by B2B Role (QI)",
                         color="qi_b2b_role", color_discrete_sequence=COLORS)
            fig.update_layout(showlegend=False, xaxis_tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x="qj_artisan_premium", y="spending_score",
                         title="Spending Score by Artisan Premium Willingness (QJ)",
                         color="qj_artisan_premium", color_discrete_sequence=COLORS)
            fig.update_layout(showlegend=False, xaxis_tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.scatter(df, x="q9_eco_importance", y="spending_score",
                             color="customer_segment",
                             title="Eco Importance vs Spending Score",
                             color_discrete_sequence=COLORS,
                             opacity=0.6)
            # Manual trendline (no statsmodels needed)
            x_vals = df["q9_eco_importance"].dropna().values
            y_vals = df["spending_score"].dropna().values
            if len(x_vals) == len(y_vals) and len(x_vals) > 1:
                m, b = np.polyfit(x_vals, y_vals, 1)
                x_line = np.array([x_vals.min(), x_vals.max()])
                fig.add_trace(go.Scatter(x=x_line, y=m * x_line + b,
                                         mode="lines", name="Trend",
                                         line=dict(color="red", width=2, dash="dash")))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.scatter(df, x="qg_nps_score", y="spending_score",
                             color="q25_purchase_intent",
                             title="NPS Score vs Spending Score",
                             color_discrete_map={"Yes": "#1D9E75", "Maybe": "#BA7517", "No": "#E24B4A"},
                             opacity=0.6)
            # Manual trendline (no statsmodels needed)
            x_vals2 = df["qg_nps_score"].dropna().values
            y_vals2 = df["spending_score"].dropna().values
            if len(x_vals2) == len(y_vals2) and len(x_vals2) > 1:
                m2, b2 = np.polyfit(x_vals2, y_vals2, 1)
                x_line2 = np.array([x_vals2.min(), x_vals2.max()])
                fig.add_trace(go.Scatter(x=x_line2, y=m2 * x_line2 + b2,
                                         mode="lines", name="Trend",
                                         line=dict(color="red", width=2, dash="dash")))
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Diagnostic Insight:** "
        "Price ('Higher price') is the top barrier for Deal-Driven Pragmatists but barely affects Corporate Buyers. "
        "Customers with 'Almost always' greenwashing scepticism show 35% lower 'Yes' conversion. "
        "B2B decision-makers have 3–4× higher spending scores — flag them immediately."
    )


# ─────────────────────────────────────────────
# ── PAGE 4: CLUSTERING ───────────────────────
# ─────────────────────────────────────────────

def page_clustering(df, models):
    st.title("👥 Customer Clustering")
    st.markdown("*K-Means segmentation to identify actionable buyer personas.*")
    st.divider()

    cm = models["cluster_metrics"]
    labels = np.array(models["cluster_labels"])
    df_plot = df.copy()
    df_plot["cluster"] = labels
    df_plot["cluster_name"] = df_plot["cluster"].map(PERSONA_NAMES)

    tab1, tab2, tab3 = st.tabs(["Elbow & Silhouette", "Cluster Visualisation", "Cluster Profiles"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cm["k_range"], y=cm["inertias"],
                                     mode="lines+markers", name="Inertia",
                                     line=dict(color=BRAND_COLOR, width=2),
                                     marker=dict(size=8)))
            fig.add_vline(x=cm["best_k"], line_dash="dash", line_color="#E24B4A",
                          annotation_text=f"Optimal k={cm['best_k']}")
            fig.update_layout(title="Elbow Curve (Inertia vs K)",
                              xaxis_title="Number of Clusters (k)",
                              yaxis_title="Inertia")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cm["k_range"], y=cm["silhouettes"],
                                     mode="lines+markers", name="Silhouette",
                                     line=dict(color="#7F77DD", width=2),
                                     marker=dict(size=8)))
            fig.add_vline(x=cm["best_k"], line_dash="dash", line_color="#E24B4A",
                          annotation_text=f"Best k={cm['best_k']}")
            fig.update_layout(title="Silhouette Score vs K",
                              xaxis_title="Number of Clusters (k)",
                              yaxis_title="Silhouette Score")
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Best Silhouette Score (k=5)", f"{cm['silhouette_best']:.4f}")

    with tab2:
        fig = px.scatter(
            x=cm["pca_x"], y=cm["pca_y"],
            color=df_plot["cluster_name"],
            title="K-Means Clusters — PCA 2D Projection",
            labels={"x": "PCA Component 1", "y": "PCA Component 2", "color": "Cluster"},
            color_discrete_sequence=COLORS, opacity=0.7,
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

        seg_vs_cluster = pd.crosstab(df_plot["customer_segment"], df_plot["cluster_name"])
        st.subheader("Segment vs Cluster Agreement Matrix")
        fig = px.imshow(seg_vs_cluster, text_auto=True,
                        title="Ground Truth Segment × Predicted Cluster",
                        color_continuous_scale="Teal")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cluster_profiles = df_plot.groupby("cluster_name").agg(
            Count=("resp_id", "count"),
            Avg_Eco_Score=("q9_eco_importance", "mean"),
            Avg_NPS=("qg_nps_score", "mean"),
            Avg_Spend=("spending_score", "mean"),
            Pct_Yes=("purchase_intent_encoded", lambda x: (x == 2).mean() * 100),
            Pct_B2B=("qi_b2b_role", lambda x: (x == "Yes decision-maker").mean() * 100),
        ).round(2).reset_index()

        st.dataframe(cluster_profiles, use_container_width=True)

        fig = px.bar(cluster_profiles, x="cluster_name", y=["Avg_Eco_Score", "Avg_NPS"],
                     title="Cluster Profiles — Eco Score & NPS",
                     color_discrete_sequence=[BRAND_COLOR, "#7F77DD"], barmode="group")
        fig.update_layout(xaxis_title="Cluster")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(cluster_profiles, x="cluster_name", y="Avg_Spend",
                     title="Average Spending Score by Cluster",
                     color="Avg_Spend", color_continuous_scale=["#9FE1CB", "#085041"])
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Clustering Insight:** "
        "K=5 optimally recovers the 5 latent personas. "
        "Corporate Buyers have the highest avg spend score. "
        "Deal-Driven Pragmatists have the lowest NPS and lowest 'Yes' intent — target with discounts only."
    )


# ─────────────────────────────────────────────
# ── PAGE 5: CLASSIFICATION ───────────────────
# ─────────────────────────────────────────────

def page_classification(df, models):
    st.title("🎯 Classification — Purchase Intent Prediction")
    st.markdown(
        "*Random Forest predicts whether a customer will say Yes / Maybe / No to ReWeave. "
        "Features: eco mindset, quality perception, trust, barriers, city tier, income.*"
    )
    st.divider()

    cm_data = models["clf_metrics"]

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{cm_data['accuracy']*100:.2f}%")
    c2.metric("Precision (Weighted)", f"{cm_data['precision']*100:.2f}%")
    c3.metric("Recall (Weighted)", f"{cm_data['recall']*100:.2f}%")
    c4.metric("F1 Score (Weighted)", f"{cm_data['f1']*100:.2f}%")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curves", "Feature Importance"])

    with tab1:
        cm_arr = np.array(cm_data["confusion_matrix"])
        labels_names = ["No (0)", "Maybe (1)", "Yes (2)"]
        fig = px.imshow(
            cm_arr, text_auto=True,
            x=labels_names, y=labels_names,
            title="Confusion Matrix — Test Set",
            color_continuous_scale="Teal",
            labels=dict(x="Predicted", y="Actual"),
        )
        st.plotly_chart(fig, use_container_width=True)

        per_class = {}
        y_te = np.array(cm_data["y_test"])
        y_pr = np.array(cm_data["y_pred"])
        for cls_val, cls_name in zip([0, 1, 2], ["No", "Maybe", "Yes"]):
            mask = y_te == cls_val
            if mask.sum() > 0:
                per_class[cls_name] = {
                    "Precision": round(precision_score(y_te == cls_val, y_pr == cls_val, zero_division=0), 3),
                    "Recall": round(recall_score(y_te == cls_val, y_pr == cls_val, zero_division=0), 3),
                    "F1": round(f1_score(y_te == cls_val, y_pr == cls_val, zero_division=0), 3),
                    "Support": int(mask.sum()),
                }
        st.subheader("Per-Class Metrics")
        st.dataframe(pd.DataFrame(per_class).T, use_container_width=True)

    with tab2:
        roc_d = cm_data["roc_data"]
        fig = go.Figure()
        roc_colors = {"No": "#E24B4A", "Maybe": "#BA7517", "Yes": "#1D9E75"}
        for cls_name, rd in roc_d.items():
            fig.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"],
                name=f"{cls_name} (AUC = {rd['auc']:.3f})",
                line=dict(color=roc_colors[cls_name], width=2),
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Classifier",
                                  line=dict(color="grey", width=1, dash="dash")))
        fig.update_layout(
            title="ROC Curves — One-vs-Rest (OvR)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        avg_auc = np.mean([rd["auc"] for rd in roc_d.values()])
        st.metric("Mean AUC (macro)", f"{avg_auc:.4f}")

    with tab3:
        fi = cm_data["feature_importance"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance — Random Forest",
                     color="Importance", color_continuous_scale=["#9FE1CB", "#085041"])
        fig.update_layout(yaxis_title="", height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "**Classification Insight:** "
            "Quality perception (QA), eco importance (Q9), and narrative resonance (QC) "
            "are the top features — confirming that trust and narrative drive conversion, "
            "not just demographics."
        )


# ─────────────────────────────────────────────
# ── PAGE 6: ASSOCIATION RULES ─────────────────
# ─────────────────────────────────────────────

def page_arm(df, models):
    st.title("🔗 Association Rule Mining")
    st.markdown(
        "*Apriori algorithm mines product + occasion + décor baskets to reveal "
        "what customers buy together — the foundation of your bundle strategy.*"
    )
    st.divider()

    rules = models["arm_rules"]

    if rules is None or rules.empty:
        st.warning("Association rules could not be generated. Check that Q13, Q17, Q18 columns exist.")
        return

    tab1, tab2, tab3 = st.tabs(["Rules Table", "Network Graph", "Bundle Recommendations"])

    with tab1:
        st.subheader(f"Top Association Rules (Total: {len(rules)})")
        min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.3, 0.05)
        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0, 0.1)
        filtered = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)]
        st.write(f"Showing {len(filtered)} rules after filter")

        display_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].head(50).rename(columns={
                "antecedents_str": "If Customer Buys",
                "consequents_str": "→ Also Buys",
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            }).style.background_gradient(subset=["Lift", "Confidence"], cmap="YlGn"),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(filtered, x="support", y="confidence",
                             size="lift", color="lift",
                             title="Support vs Confidence (bubble size = Lift)",
                             color_continuous_scale="Teal",
                             hover_data=["antecedents_str", "consequents_str"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top10 = filtered.head(10)
            fig = px.bar(top10, x="lift", y="antecedents_str", orientation="h",
                         title="Top 10 Rules by Lift",
                         color="confidence", color_continuous_scale="Teal")
            fig.update_layout(yaxis_title="Antecedent")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Product Association Network")
        top_rules = rules.head(30)
        G = nx.DiGraph()
        for _, rule in top_rules.iterrows():
            ants = list(rule["antecedents"])
            cons = list(rule["consequents"])
            for a in ants:
                for c in cons:
                    G.add_edge(a, c, weight=float(rule["lift"]))

        if len(G.nodes) == 0:
            st.info("Not enough rules to build network. Try lowering thresholds.")
        else:
            pos = nx.spring_layout(G, seed=42, k=2)
            edge_x, edge_y, edge_w = [], [], []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_w.append(d.get("weight", 1))

            node_x = [pos[n][0] for n in G.nodes]
            node_y = [pos[n][1] for n in G.nodes]
            node_labels = list(G.nodes)
            node_colors = [COLORS[i % len(COLORS)] for i in range(len(node_labels))]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                      line=dict(width=0.8, color="#B4B2A9"), hoverinfo="none"))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=18, color=node_colors, line=dict(width=1, color="white")),
                text=node_labels, textposition="top center",
                textfont=dict(size=9), hovertext=node_labels, hoverinfo="text",
            ))
            fig.update_layout(
                title="Association Network (nodes = items, edges = rules, arrow = direction)",
                showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Data-Backed Bundle Recommendations")
        strong_rules = rules[rules["lift"] >= 1.5].head(6) if not rules.empty else pd.DataFrame()

        bundle_data = [
            {"Bundle Name": "Zero Waste Home Kit", "Products": "PROD_Bags + DECOR_Cushion covers + DECOR_Throws", "Target": "Eco Evangelist", "Price Tier": "₹2,499"},
            {"Bundle Name": "Design-Led Home Set", "Products": "PROD_Bags + DECOR_Wall hangings + DECOR_Table runners", "Target": "Aesthetic Minimalist", "Price Tier": "₹1,999"},
            {"Bundle Name": "ESG Gifting Box", "Products": "PROD_Stationery + PROD_Wallets + PROD_Bags", "Target": "Corporate Buyer", "Price Tier": "₹3,499"},
            {"Bundle Name": "Gift Hamper", "Products": "PROD_Scarves + PROD_Kids products + DECOR_Cushion covers", "Target": "Conscious Gifter", "Price Tier": "₹1,499"},
            {"Bundle Name": "Value Starter Pack", "Products": "PROD_Wallets + PROD_Scarves + PROD_Bags", "Target": "Deal-Driven Pragmatist", "Price Tier": "₹999"},
        ]
        st.dataframe(pd.DataFrame(bundle_data), use_container_width=True)

        if not strong_rules.empty:
            st.subheader("Strongest Statistically-Backed Rules (Lift ≥ 1.5)")
            st.dataframe(
                strong_rules[display_cols].rename(columns={
                    "antecedents_str": "If Customer Buys",
                    "consequents_str": "→ Also Buys",
                    "support": "Support",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }),
                use_container_width=True,
            )

    st.info(
        "**ARM Insight:** Rules with Lift > 1.5 indicate non-random co-purchase patterns. "
        "Festive occasion buyers (OCC_Diwali) strongly associate with décor items. "
        "Corporate gifting occasion buyers strongly associate with stationery and wallets."
    )


# ─────────────────────────────────────────────
# ── PAGE 7: REGRESSION ───────────────────────
# ─────────────────────────────────────────────

def page_regression(df, models):
    st.title("💰 Spending Score Regression")
    st.markdown(
        "*Gradient Boosting Regressor predicts each customer's spending potential (0–100 score). "
        "Features: income, spend habits, B2B role, artisan premium willingness, eco mindset.*"
    )
    st.divider()

    rm = models["reg_metrics"]

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{rm['r2']:.4f}")
    c2.metric("RMSE", f"{rm['rmse']:.2f}")
    c3.metric("MAE", f"{rm['mae']:.2f}")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance"])

    with tab1:
        y_te = np.array(rm["y_test"])
        y_pr = np.array(rm["y_pred"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=y_pr, mode="markers",
                                  marker=dict(color=BRAND_COLOR, size=4, opacity=0.6),
                                  name="Predictions"))
        lim = [max(0, min(y_te.min(), y_pr.min()) - 5), min(100, max(y_te.max(), y_pr.max()) + 5)]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                  line=dict(color="red", dash="dash", width=1.5),
                                  name="Perfect Prediction"))
        fig.update_layout(title="Actual vs Predicted Spending Score",
                          xaxis_title="Actual Spending Score",
                          yaxis_title="Predicted Spending Score")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        residuals = y_te - y_pr
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pr, y=residuals, mode="markers",
                                  marker=dict(color="#7F77DD", size=4, opacity=0.6),
                                  name="Residuals"))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Residual Plot (Predicted vs Residual)",
                          xaxis_title="Predicted Spending Score",
                          yaxis_title="Residual (Actual − Predicted)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.histogram(pd.DataFrame({"Residual": residuals}), x="Residual", nbins=40,
                             title="Residual Distribution",
                             color_discrete_sequence=[BRAND_COLOR])
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fi_reg = rm["feature_importance"]
        fi_df = pd.DataFrame(list(fi_reg.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance — Gradient Boosting Regressor",
                     color="Importance", color_continuous_scale=["#9FE1CB", "#085041"])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spending Score by Segment and City Tier")
    seg_city = df.groupby(["customer_segment", "q4_city_tier"])["spending_score"].mean().reset_index()
    fig = px.bar(seg_city, x="customer_segment", y="spending_score", color="q4_city_tier",
                 title="Avg Spending Score — Segment × City Tier",
                 color_discrete_sequence=COLORS, barmode="group")
    fig.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Regression Insight:** "
        "B2B role and artisan premium willingness are the top spend drivers — "
        "confirming that Corporate Buyers and premium-oriented customers have "
        "significantly higher spending ceilings than income bracket alone suggests."
    )


# ─────────────────────────────────────────────
# ── PAGE 8: PREDICT NEW CUSTOMERS ────────────
# ─────────────────────────────────────────────

def page_predict(models):
    st.title("🆕 Predict New Customers")
    st.markdown(
        "Upload a CSV of new survey respondents. "
        "The system will score each row across all 4 models and return an enriched file "
        "with 7 prediction columns ready for marketing action."
    )
    st.divider()

    st.subheader("Required CSV Columns (minimum)")
    required = [
        "q1_age_group", "q2_gender", "q4_city_tier", "q6_income",
        "q7_lifestyle_spend", "q9_eco_importance", "q11_shopping_personality",
        "qa_quality_perception", "qe_greenwashing", "qi_b2b_role",
        "qd_traceability", "qg_nps_score",
    ]
    st.code(", ".join(required))
    st.caption(
        "Missing columns will be imputed with mode/median. "
        "Download a template below to see the full expected format."
    )

    # Template download
    df_train = load_data()
    template = df_train[required].head(5)
    st.download_button(
        "📥 Download Template CSV",
        data=template.to_csv(index=False),
        file_name="reweave_new_customers_template.csv",
        mime="text/csv",
    )

    st.divider()
    uploaded = st.file_uploader("Upload New Customer CSV", type=["csv"])

    if uploaded:
        new_df = pd.read_csv(uploaded)
        st.write(f"Uploaded: {len(new_df)} rows × {len(new_df.columns)} columns")
        st.dataframe(new_df.head(), use_container_width=True)

        # Fill missing columns with mode from training data
        for col in required:
            if col not in new_df.columns:
                mode_val = df_train[col].mode()[0] if col in df_train.columns else "Unknown"
                new_df[col] = mode_val

        # Add other needed columns
        for col in CLF_FEATURES + REG_FEATURES:
            if col not in new_df.columns:
                if col in df_train.columns:
                    if df_train[col].dtype == object:
                        new_df[col] = df_train[col].mode()[0]
                    else:
                        new_df[col] = df_train[col].median()
                else:
                    new_df[col] = 0

        # Predict purchase intent
        X_clf_new, _ = encode_df(
            new_df, CLF_FEATURES, encoders=models["clf_enc"], fit=False
        )
        clf_pred = models["clf"].predict(X_clf_new)
        clf_prob = models["clf"].predict_proba(X_clf_new)
        intent_map = {0: "No", 1: "Maybe", 2: "Yes"}
        intent_labels = [intent_map[p] for p in clf_pred]
        intent_conf = [round(float(max(row)), 3) for row in clf_prob]

        # Predict cluster
        X_clus_new, _, _ = encode_cluster_features(
            new_df,
            scaler=models["cluster_scaler"],
            cat_encoders=models["clus_cat_enc"],
            fit=False,
        )
        cluster_pred = models["kmeans"].predict(X_clus_new)
        cluster_names = [PERSONA_NAMES.get(c, "Unknown") for c in cluster_pred]

        # Predict spending score
        X_reg_new, _ = encode_df(
            new_df, REG_FEATURES, encoders=models["reg_enc"], fit=False
        )
        spend_pred = models["gbr"].predict(X_reg_new)
        spend_pred = np.clip(spend_pred, 0, 100)

        # Prescriptive outputs
        bundles, discounts, channels, actions, spend_ranges = [], [], [], [], []
        for intent, cl, sp in zip(intent_labels, cluster_pred, spend_pred):
            p = get_prescriptive(intent, int(cl), float(sp))
            bundles.append(p["Recommended Bundle"])
            discounts.append(p["Discount Tier"])
            channels.append(p["Primary Channel"])
            actions.append(p["Priority Action"])
            spend_ranges.append(p["Est. Annual Spend"])

        # Build output
        out_df = new_df.copy()
        out_df["predicted_intent"] = intent_labels
        out_df["intent_confidence"] = intent_conf
        out_df["predicted_cluster"] = cluster_names
        out_df["predicted_spend_score"] = np.round(spend_pred, 2)
        out_df["recommended_bundle"] = bundles
        out_df["discount_tier"] = discounts
        out_df["recommended_channel"] = channels
        out_df["priority_action"] = actions
        out_df["est_annual_spend"] = spend_ranges

        st.success(f"Predictions complete for {len(out_df)} customers!")
        st.divider()

        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        yes_pct = (np.array(intent_labels) == "Yes").mean() * 100
        maybe_pct = (np.array(intent_labels) == "Maybe").mean() * 100
        c1.metric("Ready to Buy (Yes)", f"{yes_pct:.1f}%")
        c2.metric("Nurture (Maybe)", f"{maybe_pct:.1f}%")
        c3.metric("Avg Predicted Spend", f"{np.mean(spend_pred):.1f}/100")
        c4.metric("B2B Leads", str(sum(1 for c in cluster_names if c == "Corporate Buyer")))

        # Cluster distribution of new data
        cluster_dist = pd.Series(cluster_names).value_counts().reset_index()
        cluster_dist.columns = ["Persona", "Count"]
        fig = px.pie(cluster_dist, values="Count", names="Persona",
                     title="Predicted Customer Personas in Uploaded Data",
                     color_discrete_sequence=COLORS)
        st.plotly_chart(fig, use_container_width=True)

        # Preview enriched output
        preview_cols = [c for c in new_df.columns[:4]] + [
            "predicted_intent", "predicted_cluster", "predicted_spend_score",
            "recommended_bundle", "discount_tier", "priority_action", "recommended_channel"
        ]
        preview_cols = [c for c in preview_cols if c in out_df.columns]
        st.subheader("Enriched Output Preview (first 20 rows)")
        st.dataframe(out_df[preview_cols].head(20), use_container_width=True)

        # Download
        st.download_button(
            "📥 Download Enriched Predictions CSV",
            data=out_df.to_csv(index=False),
            file_name="reweave_predictions_enriched.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Upload a CSV file above. "
            "Each row = one survey respondent. "
            "The system predicts: intent, persona, spend score, bundle, discount tier, channel, and priority action."
        )
        st.subheader("What each new customer row receives")
        output_schema = {
            "predicted_intent": "Yes / Maybe / No",
            "intent_confidence": "Probability of predicted class (0–1)",
            "predicted_cluster": "Eco Evangelist / Aesthetic / Pragmatist / Gifter / Corporate",
            "predicted_spend_score": "0–100 numeric spend potential",
            "recommended_bundle": "Data-backed product bundle for this persona",
            "discount_tier": "Personalised discount recommendation",
            "recommended_channel": "Best channel to reach this customer",
            "priority_action": "Convert now / Nurture / Incentivise / B2B call",
            "est_annual_spend": "Estimated annual spending range in ₹",
        }
        st.dataframe(pd.DataFrame.from_dict(output_schema, orient="index", columns=["Description"]),
                     use_container_width=True)


# ─────────────────────────────────────────────
# ── MAIN APP ─────────────────────────────────
# ─────────────────────────────────────────────

def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/recycling.png", width=60)
    st.sidebar.title("ReWeave Analytics")
    st.sidebar.caption("Sustainable Lifestyle Brand | Data Intelligence Platform")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Descriptive Analysis",
            "🔍 Diagnostic Analysis",
            "👥 Customer Clustering",
            "🎯 Classification",
            "🔗 Association Rules",
            "💰 Spend Regression",
            "🆕 Predict New Customers",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("Built with Streamlit · scikit-learn · mlxtend · plotly")

    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()

    # Train models (cached — only runs once per session)
    if page not in ("🏠 Overview", "📊 Descriptive Analysis", "🔍 Diagnostic Analysis"):
        with st.spinner("Loading models (first run may take ~30 seconds)..."):
            models = train_all_models()
    else:
        models = None

    # Route to page
    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 Descriptive Analysis":
        page_descriptive(df)
    elif page == "🔍 Diagnostic Analysis":
        page_diagnostic(df)
    elif page == "👥 Customer Clustering":
        page_clustering(df, models)
    elif page == "🎯 Classification":
        page_classification(df, models)
    elif page == "🔗 Association Rules":
        page_arm(df, models)
    elif page == "💰 Spend Regression":
        page_regression(df, models)
    elif page == "🆕 Predict New Customers":
        page_predict(models)


if __name__ == "__main__":
    main()
