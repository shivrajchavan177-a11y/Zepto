import os
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from seed_data import SEED_TRANSACTIONS, ITEM_CATALOG

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "transactions.csv")
ITEMS = list(ITEM_CATALOG.keys())

st.set_page_config(
    page_title="Zepto Smart Basket",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .stApp { background: #0f0b18; }

    .zepto-header {
        background: linear-gradient(120deg, #8025FB 0%, #B24BF3 50%, #FF5CB1 100%);
        padding: 22px 30px; border-radius: 18px; margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(128,37,251,0.35);
    }
    .zepto-header h1 { color: white; margin: 0; font-size: 32px; }
    .zepto-header p { color: #f0e6ff; margin: 4px 0 0 0; font-size: 15px; }

    div[data-testid="stMetric"] {
        background: #1a1526; border: 1px solid #33284a; border-radius: 14px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label { color: #c9b8e8 !important; }

    .item-card {
        background: #1a1526; border: 1px solid #33284a; border-radius: 16px;
        padding: 14px; text-align: center; transition: 0.15s;
    }
    .item-card:hover { border-color: #B24BF3; }
    .item-emoji { font-size: 40px; }
    .item-name { color: #fff; font-weight: 600; font-size: 15px; margin-top: 4px; }
    .item-price { color: #B24BF3; font-size: 13px; font-weight: 600; }

    .rec-card {
        background: linear-gradient(135deg, #1a1526, #241a35);
        border: 1px solid #B24BF3; border-radius: 14px; padding: 12px 16px;
        margin-bottom: 8px;
    }
    .rec-card b { color: #fff; }
    .rec-meta { color: #c9b8e8; font-size: 12px; }

    .cart-pill {
        background: #241a35; border-radius: 10px; padding: 8px 12px;
        margin-bottom: 6px; color: #fff; display: flex; justify-content: space-between;
    }
    section[data-testid="stSidebar"] { background: #140f1f; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# PERSISTENT TRANSACTION LOG  (this is what makes the "cart survives as data,
# but the cart widget itself resets" behaviour work)
# --------------------------------------------------------------------------------------
def _seed_file_if_missing():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        rows = []
        for i, t in enumerate(SEED_TRANSACTIONS, start=1):
            rows.append({
                "transaction_id": i,
                "items": ";".join(t),
                "placed_at": "seed-data",
                "source": "seed",
            })
        pd.DataFrame(rows).to_csv(DATA_PATH, index=False)


@st.cache_data(show_spinner=False)
def _read_log(_mtime):
    # _mtime is only used to bust the cache whenever the CSV file changes on disk
    df = pd.read_csv(DATA_PATH)
    return df


def load_transactions():
    """Always reflects what's on disk right now -> new orders persist across reruns
    and across new sessions/tabs, exactly like a real transactions table would."""
    _seed_file_if_missing()
    mtime = os.path.getmtime(DATA_PATH)
    df = _read_log(mtime).copy()
    tx_lists = df["items"].apply(lambda s: s.split(";")).tolist()
    return df, tx_lists


def append_transaction(items_with_qty: dict):
    """items_with_qty: {'Milk': 2, 'Bread': 1, ...} -> flattened with duplicates
    so a repeated item really does appear twice in the stored transaction,
    then appended as a new row to the persistent CSV log."""
    df, _ = load_transactions()
    flat_items = []
    for item, qty in items_with_qty.items():
        flat_items.extend([item] * int(qty))
    new_id = int(df["transaction_id"].max()) + 1 if len(df) else 1
    new_row = pd.DataFrame([{
        "transaction_id": new_id,
        "items": ";".join(flat_items),
        "placed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "app-order",
    }])
    updated = pd.concat([df, new_row], ignore_index=True)
    updated.to_csv(DATA_PATH, index=False)
    _read_log.clear()  # bust cache so the new row is picked up immediately
    return new_id


# --------------------------------------------------------------------------------------
# APRIORI ENGINE
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_rules(tx_lists, min_support, min_confidence):
    te = TransactionEncoder()
    encoded = te.fit(tx_lists).transform(tx_lists)
    onehot = pd.DataFrame(encoded, columns=te.columns_)

    frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame(), onehot

    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
    return frequent_itemsets, rules, onehot


def recommend_for_items(rules: pd.DataFrame, basket_items):
    """Given a set of items already in the basket, find rules whose antecedent
    is fully contained in the basket, recommend consequents not already in basket."""
    if rules.empty:
        return pd.DataFrame()
    basket_set = set(basket_items)
    mask = rules["antecedents"].apply(lambda a: set(a).issubset(basket_set))
    matches = rules[mask].copy()
    if matches.empty:
        return matches
    matches["new_items"] = matches["consequents"].apply(lambda c: set(c) - basket_set)
    matches = matches[matches["new_items"].apply(len) > 0]
    matches = matches.sort_values(["lift", "confidence"], ascending=False)
    return matches


# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
if "cart" not in st.session_state:
    st.session_state.cart = {}          # item -> qty, resets every new session
if "last_order_id" not in st.session_state:
    st.session_state.last_order_id = None

# Handle a pending cart-clear request BEFORE any qty_* widget is instantiated
# below. You cannot overwrite a widget's session_state key in the same run
# after that widget has already been created, so the actual reset is done
# here, at the top of the script, on the rerun that follows "Place Order".
if st.session_state.get("_do_clear_cart"):
    for _item in ITEMS:
        st.session_state.pop(f"qty_{_item}", None)
    st.session_state.cart = {}
    st.session_state["_do_clear_cart"] = False

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.markdown("""
<div class="zepto-header">
    <h1>🛒 Zepto Smart Basket</h1>
    <p>Live Apriori recommendation engine — every order you place becomes real training data for the next customer.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    min_support = st.slider("Minimum support", 0.02, 0.50, 0.10, 0.01,
                             help="How frequently an itemset must appear across all transactions.")
    min_confidence = st.slider("Minimum confidence", 0.05, 1.0, 0.20, 0.05,
                                help="How often the rule has held true, historically.")

    st.markdown("---")
    df_log, tx_lists = load_transactions()
    st.markdown("### 📦 Dataset")
    st.metric("Total transactions", len(tx_lists))
    new_orders = int((df_log["source"] == "app-order").sum())
    st.metric("Orders placed from this app", new_orders)

    st.markdown("---")
    if st.button("🔄 Reset to original 100 transactions", use_container_width=True):
        if os.path.exists(DATA_PATH):
            os.remove(DATA_PATH)
        _read_log.clear()
        st.session_state.cart = {}
        st.success("Dataset reset.")
        st.rerun()

# recompute engine on whatever is currently on disk
frequent_itemsets, rules, onehot = compute_rules(tx_lists, min_support, min_confidence)

# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab_shop, tab_dash, tab_rules, tab_log = st.tabs(
    ["🛍️ Shop & Get Recommendations", "📊 Dashboard", "🔗 Association Rules", "🧾 Transaction Log"]
)

# ======================================================================================
# TAB 1 — SHOP
# ======================================================================================
with tab_shop:
    left, right = st.columns([2, 1])

    with left:
        st.markdown("#### Pick your items")
        cols = st.columns(5)
        for i, item in enumerate(ITEMS):
            meta = ITEM_CATALOG[item]
            with cols[i % 5]:
                st.markdown(f"""
                <div class="item-card">
                    <div class="item-emoji">{meta['emoji']}</div>
                    <div class="item-name">{item}</div>
                    <div class="item-price">₹{meta['price']}</div>
                </div>
                """, unsafe_allow_html=True)
                qty = st.number_input(
                    "Qty", min_value=0, max_value=10,
                    value=st.session_state.cart.get(item, 0),
                    key=f"qty_{item}", label_visibility="collapsed",
                )
                if qty > 0:
                    st.session_state.cart[item] = qty
                elif item in st.session_state.cart:
                    del st.session_state.cart[item]

        st.markdown("#### 🎯 Because you're adding these, you might also like")
        if st.session_state.cart:
            recs = recommend_for_items(rules, st.session_state.cart.keys())
            if recs.empty:
                st.info("No strong cross-sell signal yet for this exact combination — add more items or lower the confidence threshold in the sidebar.")
            else:
                shown = set()
                for _, row in recs.head(6).iterrows():
                    new_items = row["new_items"] - shown
                    if not new_items:
                        continue
                    shown |= new_items
                    items_str = ", ".join(f"{ITEM_CATALOG[x]['emoji']} {x}" for x in new_items)
                    ante_str = ", ".join(row["antecedents"])
                    st.markdown(f"""
                    <div class="rec-card">
                        <b>{items_str}</b><br>
                        <span class="rec-meta">customers who bought {ante_str} also bought this ·
                        confidence {row['confidence']:.0%} · lift {row['lift']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("Add something to your basket to see live recommendations powered by the Apriori rules on the right.")

    with right:
        st.markdown("#### 🧺 Your Cart")
        if not st.session_state.cart:
            st.info("Cart is empty. Pick items on the left to get started.")
        else:
            total = 0
            for item, qty in st.session_state.cart.items():
                line_total = ITEM_CATALOG[item]["price"] * qty
                total += line_total
                st.markdown(f"""
                <div class="cart-pill">
                    <span>{ITEM_CATALOG[item]['emoji']} {item} × {qty}</span>
                    <span>₹{line_total}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"### Total: ₹{total}")

            if st.button("✅ Place Order", type="primary", use_container_width=True):
                new_id = append_transaction(st.session_state.cart)
                st.session_state.last_order_id = new_id
                st.session_state["_do_clear_cart"] = True
                st.rerun()

        if st.session_state.last_order_id:
            df_log2, tx_lists2 = load_transactions()
            st.success(
                f"Order #{st.session_state.last_order_id} placed! "
                f"Transaction log now has **{len(tx_lists2)}** transactions "
                f"(was {len(tx_lists2) - 1}). Your cart is empty and ready for the next order."
            )
            st.session_state.last_order_id = None

# ======================================================================================
# TAB 2 — DASHBOARD
# ======================================================================================
with tab_dash:
    basket_sizes = [len(t) for t in tx_lists]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", len(tx_lists))
    c2.metric("Unique items", len(ITEMS))
    c3.metric("Frequent itemsets", len(frequent_itemsets))
    c4.metric("Rules generated", len(rules))
    c5.metric("Avg basket size", f"{np.mean(basket_sizes):.1f}")

    st.markdown("---")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("##### Item popularity")
        item_counts = onehot.sum().sort_values(ascending=False)
        fig = px.bar(
            x=item_counts.values, y=item_counts.index, orientation="h",
            labels={"x": "Times purchased", "y": ""},
            color=item_counts.values, color_continuous_scale="Purples",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        st.markdown("##### Basket size distribution")
        fig2 = px.histogram(
            x=basket_sizes, nbins=max(basket_sizes) - min(basket_sizes) + 1,
            labels={"x": "Items per transaction"}, color_discrete_sequence=["#B24BF3"],
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="white", height=380, yaxis_title="Transactions")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Top frequent itemsets")
    if not frequent_itemsets.empty:
        top_fi = frequent_itemsets.head(12).copy()
        top_fi["itemset"] = top_fi["itemsets"].apply(lambda s: ", ".join(sorted(s)))
        fig3 = px.bar(top_fi, x="itemset", y="support", color="support",
                      color_continuous_scale="Purples")
        fig3.update_layout(xaxis_tickangle=45, plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                            coloraxis_showscale=False, height=420, xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No frequent itemsets at this support threshold — lower it in the sidebar.")

    st.markdown("##### Item co-occurrence heatmap")
    co = onehot.T.dot(onehot)
    np.fill_diagonal(co.values, 0)
    fig4 = px.imshow(co, color_continuous_scale="Purples", aspect="auto")
    fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white", height=450)
    st.plotly_chart(fig4, use_container_width=True)

# ======================================================================================
# TAB 3 — ASSOCIATION RULES
# ======================================================================================
with tab_rules:
    if rules.empty:
        st.warning("No rules at these thresholds — lower min support / confidence in the sidebar.")
    else:
        st.markdown("##### Rule network — which items pull each other into the basket")
        top_rules = rules.head(20)
        G = nx.DiGraph()
        for _, row in top_rules.iterrows():
            for a in row["antecedents"]:
                for c in row["consequents"]:
                    G.add_edge(a, c, weight=row["lift"])
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42, k=0.9)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#5c4a7a"),
                                     hoverinfo="none", mode="lines")
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                text=[f"{ITEM_CATALOG.get(n, {}).get('emoji','')} {n}" for n in G.nodes()],
                textposition="top center", textfont=dict(color="white", size=12),
                marker=dict(size=26, color="#B24BF3", line=dict(width=2, color="#FF5CB1")),
                hoverinfo="text",
            )
            fig5 = go.Figure(data=[edge_trace, node_trace])
            fig5.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)", height=500,
                                xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown("##### Explore recommendations for a single item")
        pick = st.selectbox("If a customer buys...", ITEMS)
        item_rules = rules[rules["antecedents"].apply(lambda x: pick in x)].sort_values("confidence", ascending=False)
        if item_rules.empty:
            st.info(f"No rule fires for {pick} at the current thresholds.")
        else:
            for _, row in item_rules.iterrows():
                items_str = ", ".join(f"{ITEM_CATALOG.get(x,{}).get('emoji','')} {x}" for x in row["consequents"])
                st.markdown(f"""
                <div class="rec-card">
                    <b>→ {items_str}</b><br>
                    <span class="rec-meta">support {row['support']:.2f} · confidence {row['confidence']:.0%} · lift {row['lift']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("##### Full rules table")
        display_rules = rules.copy()
        display_rules["antecedents"] = display_rules["antecedents"].apply(lambda s: ", ".join(s))
        display_rules["consequents"] = display_rules["consequents"].apply(lambda s: ", ".join(s))
        cols_show = ["antecedents", "consequents", "support", "confidence", "lift"]
        st.dataframe(
            display_rules[cols_show].style.format(
                {"support": "{:.2f}", "confidence": "{:.2f}", "lift": "{:.2f}"}
            ).background_gradient(cmap="Purples", subset=["lift"]),
            use_container_width=True, height=400,
        )

# ======================================================================================
# TAB 4 — TRANSACTION LOG
# ======================================================================================
with tab_log:
    st.markdown(f"##### Full transaction log ({len(df_log)} rows)")
    st.caption("Rows with source = app-order were added live by orders placed in this app. "
               "The dataset persists on disk, so it keeps growing across reruns — only your cart resets each new session.")
    show_df = df_log.copy()
    show_df["items"] = show_df["items"].str.replace(";", ", ")
    st.dataframe(show_df.sort_values("transaction_id", ascending=False),
                 use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download full transaction log (CSV)",
        data=df_log.to_csv(index=False).encode("utf-8"),
        file_name="zepto_transactions.csv", mime="text/csv",
    )
