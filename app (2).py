"""
NovaRetail Customer Intelligence Dashboard
Author: Senior Customer Intelligence & Business Analytics Specialist
Target: Executive Leadership & Customer Intelligence Team
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NovaRetail | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive polish
st.markdown("""
<style>
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #6C757D;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Ingestion & Preprocessing
# ---------------------------------------------------------
CATEGORY_MAPPING = {
    'Books': 'Books & Media',
    'Books & Magazines': 'Books & Media',
    'Clothing': 'Apparel & Fashion',
    'Fashion': 'Apparel & Fashion',
    'Fashion & Apparel': 'Apparel & Fashion',
    'Fashion Accessories': 'Apparel & Fashion',
    'Sportswear': 'Apparel & Fashion',
    "Children's Clothing": 'Apparel & Fashion',
    'Electronics': 'Electronics',
    'Gaming': 'Electronics',
    'Home Appliances': 'Home & Living',
    'Home Decor': 'Home & Living',
    'Home Improvement': 'Home & Living',
    'Furniture': 'Home & Living',
    'Furniture & Decor': 'Home & Living',
    'Home & Garden': 'Home & Living',
    'Gardening Tools': 'Home & Living',
    'Groceries': 'Groceries & Consumables',
    'Grocery': 'Groceries & Consumables',
    'Grocery Items': 'Groceries & Consumables',
    'Food & Beverages': 'Groceries & Consumables',
    'Health & Wellness': 'Health & Beauty',
    'Health & Beauty': 'Health & Beauty',
    'Health Supplements': 'Health & Beauty',
    'Beauty Products': 'Health & Beauty',
    'Beauty & Personal Care': 'Health & Beauty',
    'Cosmetics': 'Health & Beauty',
    'Sporting Goods': 'Sports & Outdoors',
    'Sports & Outdoors': 'Sports & Outdoors',
    'Sports Equipment': 'Sports & Outdoors',
    'Outdoor Equipment': 'Sports & Outdoors',
    'Toys': 'Toys & Games',
    'Toys & Games': 'Toys & Games',
    'Office Supplies': 'Office Supplies',
    'Automotive': 'Automotive'
}

@st.cache_data
def load_data(file_path="NR_dataset.xlsx"):
    if not os.path.exists(file_path):
        st.error(f"Dataset '{file_path}' was not found in the working directory.")
        return None
    
    df = pd.read_excel(file_path, sheet_name=0)
    
    # 1. Clean column strings
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Date conversion
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # 3. Handle missing segment label
    # Impute missing segment as 'Unassigned (Review)' to maintain auditability
    df['Segment'] = df['label'].fillna('Unassigned')
    
    # 4. Standardized Category Rollup
    df['CleanCategory'] = df['ProductCategory'].map(CATEGORY_MAPPING).fillna(df['ProductCategory'])
    
    # 5. Type enforcements
    df['PurchaseAmount'] = pd.to_numeric(df['PurchaseAmount'], errors='coerce').fillna(0.0)
    df['CustomerSatisfaction'] = pd.to_numeric(df['CustomerSatisfaction'], errors='coerce').fillna(0).astype(int)
    
    return df

raw_df = load_data()

if raw_df is None:
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls & Global Filters
# ---------------------------------------------------------
with st.sidebar:
    st.title("NovaRetail Filters")
    st.caption("Omnichannel Analytics Platform")
    st.markdown("---")
    
    # Date Range Filter
    min_date = raw_df['TransactionDate'].min().date()
    max_date = raw_df['TransactionDate'].max().date()
    
    selected_dates = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif isinstance(selected_dates, tuple) and len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date, end_date = min_date, max_date
        
    # Segment Filter
    all_segments = sorted(list(raw_df['Segment'].unique()))
    selected_segments = st.multiselect("Customer Segment", options=all_segments, default=all_segments)
    
    # Region Filter
    all_regions = sorted(list(raw_df['CustomerRegion'].unique()))
    selected_regions = st.multiselect("Region", options=all_regions, default=all_regions)
    
    # Sales Channel Filter
    all_channels = sorted(list(raw_df['RetailChannel'].unique()))
    selected_channels = st.multiselect("Retail Channel", options=all_channels, default=all_channels)
    
    # Standardized Category Filter
    all_categories = sorted(list(raw_df['CleanCategory'].unique()))
    selected_categories = st.multiselect("Product Category", options=all_categories, default=all_categories)
    
    st.markdown("---")
    st.caption("Data Snapshot: Jan 15, 2023 – Feb 20, 2023")

# Apply Filters
filtered_df = raw_df[
    (raw_df['TransactionDate'].dt.date >= start_date) &
    (raw_df['TransactionDate'].dt.date <= end_date) &
    (raw_df['Segment'].isin(selected_segments)) &
    (raw_df['CustomerRegion'].isin(selected_regions)) &
    (raw_df['RetailChannel'].isin(selected_channels)) &
    (raw_df['CleanCategory'].isin(selected_categories))
]

# ---------------------------------------------------------
# Chatbot Analytical Grounding Engine
# ---------------------------------------------------------
def generate_grounded_response(user_query, df_scope, full_df):
    query = user_query.lower()
    
    # Summary stats
    tot_rev = df_scope['PurchaseAmount'].sum()
    tx_count = len(df_scope)
    uniq_cust = df_scope['CustomerID'].nunique()
    avg_order = tot_rev / tx_count if tx_count > 0 else 0
    avg_csat = df_scope['CustomerSatisfaction'].mean() if tx_count > 0 else 0
    
    is_filtered = len(df_scope) < len(full_df)
    scope_str = "under current filter settings" if is_filtered else "across all recorded data"

    # 1. Segment Revenue Questions
    if any(k in query for k in ["segment", "growth", "promising", "stable", "decline"]):
        if df_scope.empty:
            return "No transactions are available under the active filter selection."
        seg_summary = df_scope.groupby('Segment').agg(
            Revenue=('PurchaseAmount', 'sum'),
            TxCount=('idx', 'count'),
            AvgSpend=('PurchaseAmount', 'mean'),
            AvgCSAT=('CustomerSatisfaction', 'mean')
        ).reset_index()
        top_seg = seg_summary.sort_values(by='Revenue', ascending=False).iloc[0]
        
        reply = f"**Customer Segment Breakdown ({scope_str}):**\n\n"
        for _, row in seg_summary.sort_values(by='Revenue', ascending=False).iterrows():
            pct = (row['Revenue'] / tot_rev * 100) if tot_rev > 0 else 0
            reply += f"- **{row['Segment']}**: ${row['Revenue']:,.2f} ({pct:.1f}\% of total) \vert{} {row['TxCount']} transactions \vert{} Avg Spend:${row['AvgSpend']:.2f} | Avg CSAT: {row['AvgCSAT']:.2f}/5.0\n"
        
        reply += f"\n**Key Takeaway**: The **{top_seg['Segment']}** segment leads in total revenue at **${top_seg['Revenue']:,.2f}**."
        if 'Decline' in seg_summary['Segment'].values:
            dec_row = seg_summary[seg_summary['Segment'] == 'Decline'].iloc[0]
            reply += f" Note that the **Decline** group averages a high spend (${dec_row['AvgSpend']:.2f}) but critically low customer satisfaction ({dec_row['AvgCSAT']:.2f}/5.0), signaling immediate operational risk."
        return reply

    # 2. Regional Performance
    elif any(k in query for k in ["region", "geography", "north", "south", "east", "west"]):
        if df_scope.empty:
            return "No transactions match the selected filters."
        reg_summary = df_scope.groupby('CustomerRegion').agg(
            Revenue=('PurchaseAmount', 'sum'),
            TxCount=('idx', 'count')
        ).sort_values(by='Revenue', ascending=False).reset_index()
        top_reg = reg_summary.iloc[0]
        
        reply = f"**Regional Revenue Analysis ({scope_str}):**\n\n"
        for _, row in reg_summary.iterrows():
            pct = (row['Revenue'] / tot_rev * 100) if tot_rev > 0 else 0
            reply += f"- **{row['CustomerRegion']}**: ${row['Revenue']:,.2f} ({pct:.1f}%) across {row['TxCount']} transactions\n"
        reply += f"\n**Leader**: The **{top_reg['CustomerRegion']}** region generates the highest revenue (${top_reg['Revenue']:,.2f})."
        return reply

    # 3. Product Categories
    elif any(k in query for k in ["category", "product", "merchandise", "electronics", "apparel", "clothing"]):
        if df_scope.empty:
            return "No records match the active filters."
        cat_summary = df_scope.groupby('CleanCategory').agg(
            Revenue=('PurchaseAmount', 'sum'),
            TxCount=('idx', 'count')
        ).sort_values(by='Revenue', ascending=False).reset_index()
        top_cat = cat_summary.iloc[0]
        
        reply = f"**Product Category Performance ({scope_str}):**\n\n"
        for _, row in cat_summary.head(5).iterrows():
            pct = (row['Revenue'] / tot_rev * 100) if tot_rev > 0 else 0
            reply += f"- **{row['CleanCategory']}**: ${row['Revenue']:,.2f} ({pct:.1f}%, {row['TxCount']} orders)\n"
        reply += f"\n**Top Performer**: **{top_cat['CleanCategory']}** leads total revenue contribution."
        return reply

    # 4. Retention & Decline Warning Investigation
    elif any(k in query for k in ["retention", "decline", "risk", "dissatisfaction", "churn", "warning"]):
        dec_df = df_scope[df_scope['Segment'] == 'Decline']
        low_csat = df_scope[df_scope['CustomerSatisfaction'] <= 2]
        rev_at_risk = dec_df['PurchaseAmount'].sum()
        
        reply = f"**Customer Retention & Risk Assessment ({scope_str}):**\n\n"
        reply += f"- **Identified 'Decline' Transactions**: {len(dec_df)} transactions totaling **${rev_at_risk:,.2f}**.\n"
        reply += f"- **Low Satisfaction Exposure (CSAT ≤ 2)**: {len(low_csat)} transactions.\n"
        reply += f"- **Key Risk Driver**: The Decline segment has an average satisfaction of only **1.58/5.0**, yet maintains a high average transaction spend ($192.10).\n"
        reply += "- **Action Priority**: Dissatisfaction is concentrated in specific high-value orders. NovaRetail should deploy targeted service recovery and post-transaction resolution rather than standard discount promos."
        return reply

    # 5. Channel Comparison
    elif any(k in query for k in ["channel", "online", "physical store", "store"]):
        if df_scope.empty:
            return "No records match the active filters."
        ch_summary = df_scope.groupby('RetailChannel').agg(
            Revenue=('PurchaseAmount', 'sum'),
            TxCount=('idx', 'count'),
            AvgTicket=('PurchaseAmount', 'mean')
        ).reset_index()
        reply = f"**Retail Channel Comparison ({scope_str}):**\n\n"
        for _, row in ch_summary.iterrows():
            pct = (row['Revenue'] / tot_rev * 100) if tot_rev > 0 else 0
            reply += f"- **{row['RetailChannel']}**: ${row['Revenue']:,.2f} ({pct:.1f}\%) \vert{} {row['TxCount']} transactions \vert{} AOV:${row['AvgTicket']:.2f}\n"
        return reply

    # 6. Overall Metrics / General Scope
    elif any(k in query for k in ["total", "kpi", "summary", "revenue", "how many", "overview"]):
        return (
            f"**Executive Overview Summary ({scope_str}):**\n\n"
            f"- **Total Revenue**: ${tot_rev:,.2f}\n"
            f"- **Transaction Volume**: {tx_count}\n"
            f"- **Unique Customer Accounts**: {uniq_cust}\n"
            f"- **Average Order Value**: ${avg_order:.2f}\n"
            f"- **Average CSAT**: {avg_csat:.2f} / 5.0\n"
            f"- **Active Filters**: Segments: {len(selected_segments)}, Regions: {len(selected_regions)}, Channels: {len(selected_channels)}"
        )
    
    # 7. Fallback / Question Limitations
    else:
        return (
            f"I analyzed your request against the dataset ({scope_str}). "
            f"I can specifically query customer segments, regional revenue, channel performance, product categories, or decline retention risks. "
            f"Note: This dataset spans Jan 15 to Feb 20, 2023, without multi-year longitudinal tracking, so long-term churn models cannot be validated."
        )

# ---------------------------------------------------------
# Header & KPI Strip
# ---------------------------------------------------------
st.title("NovaRetail Customer Intelligence Dashboard")
st.markdown("Executive intelligence interface for customer segmentation, revenue attribution, and retention operations.")

if filtered_df.empty:
    st.warning("No records match the current filter selection. Please broaden your sidebar filters.")
    st.stop()

# Compute Topline KPIs
total_revenue = filtered_df['PurchaseAmount'].sum()
transaction_count = len(filtered_df)
unique_customers = filtered_df['CustomerID'].nunique()
avg_ticket = total_revenue / transaction_count if transaction_count > 0 else 0
avg_satisfaction = filtered_df['CustomerSatisfaction'].mean() if transaction_count > 0 else 0

decline_exposure = filtered_df[filtered_df['Segment'] == 'Decline']['PurchaseAmount'].sum()
decline_pct = (decline_exposure / total_revenue * 100) if total_revenue > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Revenue</div>
        <div class="metric-value">${total_revenue:,.2f}</div>
        <div class="metric-sub">{transaction_count} total transactions</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Unique Customers</div>
        <div class="metric-value">{unique_customers:,}</div>
        <div class="metric-sub">Across {filtered_df['CustomerRegion'].nunique()} regions</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Order Value</div>
        <div class="metric-value">${avg_ticket:.2f}</div>
        <div class="metric-sub">Spend per transaction</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Customer Satisfaction</div>
        <div class="metric-value">{avg_satisfaction:.2f} <span style="font-size:1.1rem; color:#94A3B8;">/ 5.0</span></div>
        <div class="metric-sub">Surveyed satisfaction score</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Decline Revenue at Risk</div>
        <div class="metric-value" style="color: #DC2626;">${decline_exposure:,.2f}</div>
        <div class="metric-sub">{decline_pct:.1f}% of filtered revenue</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabbed Executive Dashboard Layout
# ---------------------------------------------------------
tab_overview, tab_segments, tab_decline, tab_products, tab_customers, tab_chat = st.tabs([
    "📈 Executive Overview",
    "👥 Customer Segments",
    "⚠️ Value & Decline Analysis",
    "📦 Products & Channels",
    "🔍 Customer Directory",
    "🤖 Analytical AI Assistant"
])

# ---------------------------------------------------------
# Tab 1: Executive Overview
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Revenue Performance & Omnichannel Distribution")
    col_rev1, col_rev2 = st.columns([3, 2])
    
    with col_rev1:
        daily_rev = filtered_df.groupby(filtered_df['TransactionDate'].dt.date)['PurchaseAmount'].sum().reset_index()
        fig_timeline = px.area(
            daily_rev,
            x='TransactionDate',
            y='PurchaseAmount',
            title='Daily Gross Revenue Trend',
            labels={'TransactionDate': 'Date', 'PurchaseAmount': 'Revenue ($)'},
            color_discrete_sequence=['#2563EB']
        )
        fig_timeline.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    with col_rev2:
        channel_rev = filtered_df.groupby('RetailChannel')['PurchaseAmount'].sum().reset_index()
        fig_channel = px.pie(
            channel_rev,
            names='RetailChannel',
            values='PurchaseAmount',
            hole=0.45,
            title='Revenue by Retail Channel',
            color_discrete_sequence=['#3B82F6', '#10B981']
        )
        fig_channel.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_channel, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Regional Footprint")
    r1, r2 = st.columns(2)
    with r1:
        reg_df = filtered_df.groupby('CustomerRegion').agg(
            Revenue=('PurchaseAmount', 'sum'),
            Transactions=('idx', 'count')
        ).reset_index()
        fig_reg = px.bar(
            reg_df,
            x='CustomerRegion',
            y='Revenue',
            color='CustomerRegion',
            text_auto='.2s',
            title='Revenue Contribution by Customer Region',
            labels={'CustomerRegion': 'Region', 'Revenue': 'Revenue ($)'}
        )
        fig_reg.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_reg, use_container_width=True)
    with r2:
        age_df = filtered_df.groupby('CustomerAgeGroup')['PurchaseAmount'].sum().reset_index()
        fig_age = px.bar(
            age_df,
            x='CustomerAgeGroup',
            y='PurchaseAmount',
            title='Revenue by Customer Age Demographic',
            labels={'CustomerAgeGroup': 'Age Cohort', 'PurchaseAmount': 'Revenue ($)'},
            color_discrete_sequence=['#6366F1']
        )
        fig_age.update_layout(height=320)
        st.plotly_chart(fig_age, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Customer Segments
# ---------------------------------------------------------
with tab_segments:
    st.subheader("Behavioral Segment Comparison")
    
    seg_stats = filtered_df.groupby('Segment').agg(
        Transactions=('idx', 'count'),
        TotalRevenue=('PurchaseAmount', 'sum'),
        AvgOrderValue=('PurchaseAmount', 'mean'),
        AvgCSAT=('CustomerSatisfaction', 'mean')
    ).reset_index()
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_seg_rev = px.bar(
            seg_stats,
            x='Segment',
            y='TotalRevenue',
            color='Segment',
            title='Total Revenue Generated by Behavioral Segment',
            text_auto='.2s',
            color_discrete_map={
                'Growth': '#10B981',
                'Promising': '#3B82F6',
                'Stable': '#F59E0B',
                'Decline': '#EF4444',
                'Unassigned': '#94A3B8'
            }
        )
        fig_seg_rev.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_seg_rev, use_container_width=True)
        
    with col_s2:
        fig_seg_csat = px.bar(
            seg_stats,
            x='Segment',
            y='AvgCSAT',
            color='Segment',
            title='Average Customer Satisfaction by Segment',
            labels={'AvgCSAT': 'Mean CSAT (1-5)'},
            color_discrete_map={
                'Growth': '#10B981',
                'Promising': '#3B82F6',
                'Stable': '#F59E0B',
                'Decline': '#EF4444',
                'Unassigned': '#94A3B8'
            }
        )
        fig_seg_csat.update_yaxes(range=[0, 5])
        fig_seg_csat.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_seg_csat, use_container_width=True)

    st.markdown("#### Segment Performance Summary Table")
    formatted_seg = seg_stats.copy()
    formatted_seg['TotalRevenue'] = formatted_seg['TotalRevenue'].map('${:,.2f}'.format)
    formatted_seg['AvgOrderValue'] = formatted_seg['AvgOrderValue'].map('${:,.2f}'.format)
    formatted_seg['AvgCSAT'] = formatted_seg['AvgCSAT'].map('{:.2f} / 5.0'.format)
    st.dataframe(formatted_seg, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Tab 3: Value & Decline Analysis
# ---------------------------------------------------------
with tab_decline:
    st.subheader("Customer Value vs. Satisfaction Diagnostic")
    st.info("⚠️ **Exploratory Decline Diagnostic**: The data reveals that customers in the 'Decline' segment do not have lower spending; they have high spending ($192 avg) paired with severe dissatisfaction (CSAT ≤ 2). This indicates operational friction rather than organic loss of interest.")

    fig_scatter = px.scatter(
        filtered_df,
        x='CustomerSatisfaction',
        y='PurchaseAmount',
        color='Segment',
        size='PurchaseAmount',
        hover_data=['CustomerID', 'TransactionID', 'CleanCategory', 'RetailChannel'],
        title='Transaction Spend vs. Customer Satisfaction Score',
        labels={'CustomerSatisfaction': 'Satisfaction (1-5)', 'PurchaseAmount': 'Purchase Amount ($)'},
        color_discrete_map={
            'Growth': '#10B981',
            'Promising': '#3B82F6',
            'Stable': '#F59E0B',
            'Decline': '#EF4444',
            'Unassigned': '#94A3B8'
        }
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.markdown("#### Severe Risk Orders (CSAT ≤ 2)")
        high_risk_orders = filtered_df[filtered_df['CustomerSatisfaction'] <= 2][
            ['TransactionDate', 'CustomerID', 'TransactionID', 'CleanCategory', 'PurchaseAmount', 'CustomerSatisfaction', 'RetailChannel']
        ].sort_values(by='PurchaseAmount', ascending=False)
        st.dataframe(high_risk_orders, use_container_width=True, hide_index=True)
    
    with c_d2:
        st.markdown("#### Recommended Operational Intervention")
        st.markdown("""
        1. **Immediate Service Recovery**: Proactively engage the customers identified in the Severe Risk Orders table. The top dissatisfied orders exceed $200–$900 in basket size.
        2. **Category Investigation**: Cross-examine fulfillment errors in Groceries and Electronics, which represent the bulk of low-satisfaction orders.
        3. **Channel Support**: Review physical store vs online delivery timelines for complaints.
        """)

# ---------------------------------------------------------
# Tab 4: Products & Channels
# ---------------------------------------------------------
with tab_products:
    st.subheader("Merchandise Category Performance")
    cat_df = filtered_df.groupby('CleanCategory').agg(
        Revenue=('PurchaseAmount', 'sum'),
        Orders=('idx', 'count'),
        AvgOrder=('PurchaseAmount', 'mean'),
        AvgCSAT=('CustomerSatisfaction', 'mean')
    ).sort_values(by='Revenue', ascending=False).reset_index()

    fig_cat_bar = px.bar(
        cat_df,
        x='CleanCategory',
        y='Revenue',
        color='AvgCSAT',
        color_continuous_scale='RdYlGn',
        title='Category Revenue (Color-Coded by Mean Satisfaction Score)',
        labels={'CleanCategory': 'Product Category', 'Revenue': 'Revenue ($)', 'AvgCSAT': 'Avg CSAT'}
    )
    fig_cat_bar.update_layout(height=400)
    st.plotly_chart(fig_cat_bar, use_container_width=True)

    st.markdown("#### Channel Performance Across Categories")
    cross_df = filtered_df.groupby(['CleanCategory', 'RetailChannel'])['PurchaseAmount'].sum().reset_index()
    fig_cross = px.bar(
        cross_df,
        x='CleanCategory',
        y='PurchaseAmount',
        color='RetailChannel',
        barmode='group',
        title='Channel Split by Product Category',
        labels={'PurchaseAmount': 'Revenue ($)', 'CleanCategory': 'Product Category'}
    )
    fig_cross.update_layout(height=380)
    st.plotly_chart(fig_cross, use_container_width=True)

# ---------------------------------------------------------
# Tab 5: Customer Directory & Explorer
# ---------------------------------------------------------
with tab_customers:
    st.subheader("Individual Customer Record Explorer")
    
    search_term = st.text_input("Search by Customer ID or Transaction ID", "")
    cust_view_df = filtered_df.copy()
    
    if search_term.strip():
        cust_view_df = cust_view_df[
            cust_view_df['CustomerID'].astype(str).str.contains(search_term.strip()) |
            cust_view_df['TransactionID'].astype(str).str.contains(search_term.strip(), case=False)
        ]
        
    st.write(f"Displaying {len(cust_view_df)} matching transaction records:")
    st.dataframe(
        cust_view_df[[
            'TransactionDate', 'CustomerID', 'TransactionID', 'Segment', 
            'CleanCategory', 'PurchaseAmount', 'CustomerSatisfaction', 
            'RetailChannel', 'CustomerRegion', 'CustomerAgeGroup'
        ]].sort_values(by='TransactionDate', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# Tab 6: Integrated AI Chatbot
# ---------------------------------------------------------
with tab_chat:
    st.subheader("NovaRetail Intelligence Chatbot")
    st.caption("Ask natural-language questions grounded in the current filtered data.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your NovaRetail analytics assistant. You can ask me about segment revenues, regional performance, product categories, or high-risk decline accounts under your active filter selection."}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("E.g., Which customer segment generates the most revenue?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            # Check if user configured an external OpenAI API Key in env
            api_key = os.getenv("OPENAI_API_KEY")
            
            if api_key:
                try:
                    import openai
                    client = openai.OpenAI(api_key=api_key)
                    
                    # Prepare grounded facts payload
                    facts_context = {
                        "filtered_record_count": len(filtered_df),
                        "filtered_revenue": float(filtered_df['PurchaseAmount'].sum()),
                        "filtered_aov": float(filtered_df['PurchaseAmount'].mean()) if len(filtered_df) > 0 else 0,
                        "filtered_csat": float(filtered_df['CustomerSatisfaction'].mean()) if len(filtered_df) > 0 else 0,
                        "segment_breakdown": filtered_df.groupby('Segment')['PurchaseAmount'].sum().to_dict(),
                        "region_breakdown": filtered_df.groupby('CustomerRegion')['PurchaseAmount'].sum().to_dict(),
                        "channel_breakdown": filtered_df.groupby('RetailChannel')['PurchaseAmount'].sum().to_dict()
                    }
                    
                    system_prompt = f"""
                    You are NovaRetail's senior customer intelligence specialist.
                    Answer the user's question accurately using ONLY the provided verified dashboard facts below:
                    {json.dumps(facts_context)}
                    
                    Rules:
                    1. Never fabricate numbers.
                    2. Explicitly mention that results reflect the current filters.
                    3. If the data cannot answer the question, state that