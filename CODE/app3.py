import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pinecone
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sqlite3 import Connection
import joblib
import sqlite3
import tempfile
import openai
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import altair as alt
import json
from prophet import Prophet
import networkx as nx
from pyvis.network import Network
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Pharma Procurement Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys (For demo only - in production use secrets management)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
# Load data and models (cached for performance)
@st.cache_data
def load_data():
    try:
        # Load the pharmaceutical pricing data
        df = pd.read_csv("pharma_price_benchmarking_merged.csv")
        
        # Convert date columns to datetime
        date_cols = ['Price_Source_Timestamp', 'Internal_Inventory_Date', 'Internal_Contract_Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean numeric columns
        numeric_cols = ['Unit_Price_Latest', 'Benchmark_Price', 'PO_Amount', 'Portal_Price',
                       'Internal_Inventory_Price', 'Internal_Contract_Price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    try:
        df = load_data()
        if df.empty:
            return None, None, None, None
            
        processed_df = preprocess_data(df)
        
        if processed_df is None or processed_df.empty:
            return None, None, None, None
            
        n_components = min(8, processed_df.shape[1])
        scaler = StandardScaler().fit(processed_df)
        pca = PCA(n_components=n_components).fit(scaler.transform(processed_df))
        llm_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Try to load XGBoost model if available
        xgb_model = None
        try:
            import xgboost
            if os.path.exists(r"D:\Procurement\best_xgboost_model3 (1).pkl"):
                xgb_model = joblib.load(r"D:\Procurement\best_xgboost_model3 (1).pkl")
        except Exception as e:
            st.warning(f"XGBoost model disabled: {str(e)}")
            
        return scaler, pca, llm_model, xgb_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# def init_pinecone():
#     try:
#         pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
#         index_name = "pharma-pricing"
        
#         if index_name not in pc.list_indexes().names():
#             pc.create_index(
#                 name=index_name,
#                 dimension=384,
#                 metric="cosine"
#             )
            
#         return pc.Index(index_name)
#     except Exception as e:
#         st.error(f"Error connecting to Pinecone: {e}")
#         return None
def init_pinecone():
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index_name = "pharma-pricing"
        
        if index_name not in pc.list_indexes().names():
            # Create index with the required spec parameter
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud='aws',  # or 'gcp'
                    region='us-east-1'  # choose your preferred region
                )
            )
            
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        return None

def preprocess_data(df):
    try:
        # Extract temporal features
        if 'Price_Source_Timestamp' in df.columns:
            df['price_year'] = df['Price_Source_Timestamp'].dt.year
            df['price_month'] = df['Price_Source_Timestamp'].dt.month
            df['price_day'] = df['Price_Source_Timestamp'].dt.day
        
        # Handle categorical columns
        cat_cols = ['Material_Type', 'Vendor_Name', 'GMP_Compliance', 'Specification', 
                   'Form', 'Material_Grade', 'Currency', 'Portal_Currency']
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Ensure numeric columns
        num_cols = ['Unit_Price_Latest', 'Benchmark_Price', 'Price_Deviation (%)', 
                   'Quantity_Ordered', 'PO_Amount', 'Portal_Price',
                   'Portal_vs_Unit_Deviation (%)', 'Internal_Inventory_Price',
                   'Internal_Contract_Price', 'Inventory_vs_Latest (%)',
                   'Contract_vs_Latest (%)']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Select features for model
        features = ['Material_Type', 'Vendor_Name', 'GMP_Compliance', 'Specification',
                   'Form', 'Material_Grade', 'Unit_Price_Latest', 'Benchmark_Price',
                   'Price_Deviation (%)', 'Quantity_Ordered', 'PO_Amount',
                   'price_year', 'price_month']
        
        # Only keep columns that exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        return df[features].dropna()
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def create_embeddings(row, scaler, pca, llm_model):
    try:
        # Scale and transform numerical features
        scaled_data = scaler.transform(row.values.reshape(1, -1))
        trad_embed = pca.transform(scaled_data)[0]
        
        # Create text description for LLM embedding
        text_data = ' '.join([f"{col}_{val}" for col, val in row.items()])
        llm_embed = llm_model.encode([text_data])[0]
        
        # Combine embeddings
        return np.concatenate([trad_embed, llm_embed])
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def detect_anomalies(df):
    try:
        if len(df) < 10:
            return pd.Series([False] * len(df))
            
        model = IsolationForest(contamination=0.05, random_state=42)
        anomalies = model.fit_predict(df[['Unit_Price_Latest', 'Price_Deviation (%)']].fillna(0))
        return anomalies == -1
    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")
        return pd.Series([False] * len(df))

def vendor_performance(df):
    try:
        if 'Vendor_Name' not in df.columns or 'PO_Amount' not in df.columns:
            return pd.DataFrame()
            
        perf = df.groupby('Vendor_Name').agg({
            'PO_Amount': ['count', 'sum', 'mean'],
            'Price_Source_Timestamp': lambda x: (datetime.now() - x.max()).days
        })
        perf.columns = ['PO Count', 'Total Spend', 'Avg PO Value', 'Days Since Last PO']
        return perf.sort_values('Total Spend', ascending=False)
    except Exception as e:
        st.error(f"Vendor performance calculation failed: {e}")
        return pd.DataFrame()

def forecast_spend(df):
    try:
        if len(df) < 12:
            return None
            
        ts = df.resample('M', on='Price_Source_Timestamp')['PO_Amount'].sum().reset_index()
        ts.columns = ['ds', 'y']
        
        model = Prophet()
        model.fit(ts)
        future = model.make_future_dataframe(periods=6, freq='M')
        return model.predict(future)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
        return None

def create_supplier_network(df):
    st.header("🔗 Supplier Network Analysis")
    
    if 'Vendor_Name' not in df.columns or 'Material_Name' not in df.columns:
        st.warning("Supplier network cannot be displayed - missing vendor or material data")
        return
    
    with st.expander("⚙️ Network Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_transactions = st.number_input("Minimum transactions to include", 
                                            min_value=1, 
                                            max_value=100, 
                                            value=3)
            physics_enabled = st.checkbox("Enable physics simulation", value=True)
        with col2:
            edge_threshold = st.slider("Edge weight threshold", 
                                      min_value=0, 
                                      max_value=100, 
                                      value=10)
            show_labels = st.checkbox("Show all labels", value=False)
    
    with st.spinner("Analyzing supplier relationships..."):
        try:
            G = nx.Graph()
            vendor_counts = df['Vendor_Name'].value_counts()
            material_counts = df['Material_Name'].value_counts()
            
            for _, row in df.iterrows():
                if (pd.notna(row['Vendor_Name']) and pd.notna(row['Material_Name']) and 
                   (vendor_counts[row['Vendor_Name']] >= min_transactions)):
                    if G.has_edge(row['Vendor_Name'], row['Material_Name']):
                        G[row['Vendor_Name']][row['Material_Name']]['weight'] += 1
                    else:
                        G.add_edge(row['Vendor_Name'], row['Material_Name'], weight=1)
            
            if len(G.nodes()) == 0:
                st.warning("No supplier relationships meet the current filters")
                return
            
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            
            for node in G.nodes():
                if node in vendor_counts.index:
                    size = vendor_counts.get(node, 1)
                    title = f"Vendor: {node}<br>Transactions: {size}"
                    group = 2
                else:
                    size = material_counts.get(node, 1)
                    title = f"Material: {node}<br>Orders: {size}"
                    group = 1
                
                net.add_node(node, 
                           title=title,
                           size=size*0.5 if size < 50 else 25,
                           group=group,
                           label=str(node) if show_labels else None)
            
            for edge in G.edges(data=True):
                if edge[2]['weight'] >= edge_threshold:
                    net.add_edge(edge[0], edge[1], 
                                value=edge[2]['weight'],
                                title=f"Transactions: {edge[2]['weight']}")
            
            net.set_options("""
            {
              "physics": {
                "enabled": %s,
                "barnesHut": {
                  "gravitationalConstant": -2000,
                  "centralGravity": 0.3,
                  "springLength": 150,
                  "damping": 0.09
                },
                "minVelocity": 0.75
              },
              "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 3
              }
            }
            """ % str(physics_enabled).lower())
            
            net.save_graph("network.html")
            st.components.v1.html(open("network.html").read(), height=600)
            
            st.subheader("📊 Network Metrics")
            tab1, tab2, tab3 = st.tabs(["Key Players", "Vendor Importance", "Community Structure"])
            
            with tab1:
                st.write("**Most Connected Nodes (Degree Centrality)**")
                degree_df = pd.DataFrame.from_dict(degree_centrality, 
                                                orient='index', 
                                                columns=['Centrality'])\
                                       .sort_values('Centrality', ascending=False)
                st.dataframe(degree_df.head(10).style.format({'Centrality': '{:.2%}'}))
                
                fig = px.bar(degree_df.head(10), 
                            title="Top 10 Nodes by Degree Centrality")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.write("**Bridge Nodes (Betweenness Centrality)**")
                between_df = pd.DataFrame.from_dict(betweenness_centrality, 
                                                  orient='index', 
                                                  columns=['Centrality'])\
                                       .sort_values('Centrality', ascending=False)
                st.dataframe(between_df.head(10).style.format({'Centrality': '{:.2%}'}))
                
                fig = px.bar(between_df.head(10), 
                            title="Top 10 Bridge Nodes in Network")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                try:
                    communities = nx.algorithms.community.greedy_modularity_communities(G)
                    st.write(f"**Detected {len(communities)} Communities**")
                    
                    for i, comm in enumerate(communities):
                        st.write(f"Community {i+1}: {len(comm)} members")
                        if st.checkbox(f"Show members for Community {i+1}", key=f"comm_{i}"):
                            st.write(list(comm))
                    
                    for node in G.nodes():
                        for i, comm in enumerate(communities):
                            if node in comm:
                                G.nodes[node]['group'] = i + 1
                    
                    net_comm = Network(height="500px", width="100%", notebook=True)
                    net_comm.from_nx(G)
                    net_comm.save_graph("network_comm.html")
                    st.components.v1.html(open("network_comm.html").read(), height=500)
                    
                except Exception as e:
                    st.warning(f"Community detection failed: {str(e)}")
            
            st.subheader("🔗 Relationship Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Nodes", len(G.nodes()))
            col2.metric("Total Relationships", len(G.edges()))
            
            if nx.is_connected(G):
                diameter = nx.diameter(G)
                col3.metric("Network Diameter", diameter)
            else:
                col3.metric("Connected Components", nx.number_connected_components(G))
            
            st.write("**Vendor-Material Transaction Matrix**")
            vendor_material = df.groupby(['Vendor_Name', 'Material_Name']).size().unstack().fillna(0)
            st.dataframe(vendor_material.style.background_gradient(cmap='Blues'))
            
        except Exception as e:
            st.error(f"Network analysis failed: {str(e)}")

def ai_powered_search_section(df, processed_df, scaler, pca, llm_model, pinecone_index):
    st.header("🔍 AI-Powered Pharma Pricing Search")
    
    with st.expander("🔎 Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Minimum Unit Price", 
                                     value=float(df['Unit_Price_Latest'].min()),
                                     key="min_price")
            max_price = st.number_input("Maximum Unit Price",
                                     value=float(df['Unit_Price_Latest'].max()),
                                     key="max_price")
            date_range = st.date_input("Date Range",
                                    [df['Price_Source_Timestamp'].min(), df['Price_Source_Timestamp'].max()],
                                    key="date_range")
        with col2:
            material_types = st.multiselect("Material Types", 
                                         df['Material_Type'].unique(),
                                         key="material_types")
            vendors = st.multiselect("Vendors", df['Vendor_Name'].unique(),
                                   key="vendors")
    
    filtered_df = df.copy()
    if min_price is not None and max_price is not None:
        filtered_df = filtered_df[(filtered_df['Unit_Price_Latest'] >= min_price) & 
                                 (filtered_df['Unit_Price_Latest'] <= max_price)]
    if date_range and len(date_range) == 2:
        filtered_df = filtered_df[(filtered_df['Price_Source_Timestamp'] >= pd.to_datetime(date_range[0])) & 
                                 (filtered_df['Price_Source_Timestamp'] <= pd.to_datetime(date_range[1]))]
    if material_types:
        filtered_df = filtered_df[filtered_df['Material_Type'].isin(material_types)]
    if vendors:
        filtered_df = filtered_df[filtered_df['Vendor_Name'].isin(vendors)]
    
    st.subheader("Find Similar Material Pricing")
    material_name = st.selectbox("Select a Material to find similar pricing", 
                               filtered_df['Material_Name'].unique(),
                               key="material_select")
    
    if st.button("Find Similar Pricing", key="find_similar"):
        with st.spinner("Analyzing pricing patterns..."):
            try:
                # Get the selected material data
                material_data = filtered_df[filtered_df['Material_Name'] == material_name]
                if material_data.empty:
                    st.error(f"No data found for material: {material_name}")
                    return
                
                material_data = material_data.iloc[0]
                
                # Create embedding
                if material_data.name not in processed_df.index:
                    st.error("Selected material not found in processed data")
                    return
                
                embedding = create_embeddings(
                    processed_df.loc[material_data.name], 
                    scaler, 
                    pca, 
                    llm_model
                )
                
                if embedding is None:
                    st.error("Failed to create embedding for the selected material")
                    return
                
                # Build filters
                filter_dict = {}
                if min_price is not None:
                    filter_dict["Unit_Price_Latest"] = {"$gte": float(min_price)}
                if max_price is not None:
                    if "Unit_Price_Latest" in filter_dict:
                        filter_dict["Unit_Price_Latest"]["$lte"] = float(max_price)
                    else:
                        filter_dict["Unit_Price_Latest"] = {"$lte": float(max_price)}
                
                if date_range and len(date_range) == 2:
                    start_timestamp = datetime.combine(date_range[0], datetime.min.time()).timestamp()
                    end_timestamp = datetime.combine(date_range[1], datetime.max.time()).timestamp()
                    filter_dict["Price_Source_Timestamp"] = {
                        "$gte": start_timestamp,
                        "$lte": end_timestamp
                    }
                
                # First try with filters
                results = pinecone_index.query(
                    vector=embedding.tolist(),
                    top_k=10,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
                
                # If no results, try without filters
                if not results or 'matches' not in results or len(results['matches']) == 0:
                    st.warning("No results with current filters, trying without filters...")
                    results = pinecone_index.query(
                        vector=embedding.tolist(),
                        top_k=10,
                        include_metadata=True
                    )
                
                if not results or 'matches' not in results or len(results['matches']) == 0:
                    st.warning("No similar pricing found in our database")
                    return
                
                st.success(f"Found {len(results['matches'])} similar pricing records")
                
                # Display results
                valid_matches = 0
                for match in results['matches']:
                    try:
                        # Get the material by ID from the original DataFrame
                        # First try to find by index (if IDs are indices)
                        try:
                            match_id = int(match['id'])
                            pricing_details = df.iloc[match_id]
                        except:
                            # If not found by index, try to find by Material_Name
                            pricing_details = df[df['Material_Name'] == match['id']].iloc[0]
                        
                        with st.expander(f"Material: {pricing_details['Material_Name']} | Similarity: {match['score']:.2f}"):
                            cols = st.columns(3)
                            cols[0].metric("Unit Price", f"{pricing_details['Unit_Price_Latest']:,.2f}")
                            cols[1].metric("Vendor", pricing_details.get('Vendor_Name', 'N/A'))
                            
                            price_date = pricing_details['Price_Source_Timestamp']
                            if isinstance(price_date, (int, float)):
                                price_date = datetime.fromtimestamp(price_date).strftime('%Y-%m-%d')
                            else:
                                price_date = price_date.strftime('%Y-%m-%d') if hasattr(price_date, 'strftime') else str(price_date)
                            
                            cols[2].metric("Date", price_date)
                            
                            comparison_df = pd.DataFrame({
                                'Attribute': ['Material Type', 'Vendor', 'Specification', 'Grade', 'Price'],
                                'Selected Material': [
                                    material_data.get('Material_Type', 'N/A'),
                                    material_data.get('Vendor_Name', 'N/A'),
                                    material_data.get('Specification', 'N/A'),
                                    material_data.get('Material_Grade', 'N/A'),
                                    f"${material_data.get('Unit_Price_Latest', 0):,.2f}"
                                ],
                                'Similar Material': [
                                    pricing_details.get('Material_Type', 'N/A'),
                                    pricing_details.get('Vendor_Name', 'N/A'),
                                    pricing_details.get('Specification', 'N/A'),
                                    pricing_details.get('Material_Grade', 'N/A'),
                                    f"${pricing_details.get('Unit_Price_Latest', 0):,.2f}"
                                ]
                            })
                            st.dataframe(comparison_df, hide_index=True)
                            
                            # Add price difference analysis
                            price_diff = pricing_details.get('Unit_Price_Latest', 0) - material_data.get('Unit_Price_Latest', 0)
                            price_diff_pct = (price_diff / material_data.get('Unit_Price_Latest', 1)) * 100 if material_data.get('Unit_Price_Latest', 0) != 0 else 0
                            
                            st.metric("Price Difference", 
                                     f"${abs(price_diff):,.2f}",
                                     delta=f"{price_diff_pct:.1f}% {'higher' if price_diff > 0 else 'lower'}",
                                     delta_color="inverse")
                        
                        valid_matches += 1
                    
                    except Exception as e:
                        st.warning(f"Skipping match {match['id']}: {str(e)}")
                        continue
                
                if valid_matches == 0:
                    st.warning("No valid matches could be displayed")
                    st.info("Showing filtered results instead:")
                    st.dataframe(filtered_df[['Material_Name', 'Vendor_Name', 'Unit_Price_Latest', 
                                            'Price_Source_Timestamp']].head(10))
            
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                st.exception(e)

def procurement_dashboard(df):
    st.header("📊 Pharma Pricing Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spend", f"${df['PO_Amount'].sum():,.0f}" if not df.empty else "$0")
    with col2:
        st.metric("Avg Unit Price", f"${df['Unit_Price_Latest'].mean():,.2f}" if not df.empty else "$0")
    with col3:
        st.metric("Vendor Count", df['Vendor_Name'].nunique() if 'Vendor_Name' in df.columns else "N/A")
    with col4:
        st.metric("Material Count", df['Material_Name'].nunique() if 'Material_Name' in df.columns else "N/A")
    
    tab1, tab2, tab3 = st.tabs(["Price Trend", "Vendor Distribution", "Price Anomalies"])
    
    with tab1:
        if not df.empty:
            monthly = df.resample('M', on='Price_Source_Timestamp')['Unit_Price_Latest'].mean()
            fig = px.line(monthly, title="Average Monthly Unit Price Trend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for trend analysis")
            
    with tab2:
        if not df.empty and 'Vendor_Name' in df.columns:
            vendor_price = df.groupby('Vendor_Name')['Unit_Price_Latest'].mean().nlargest(10)
            fig = px.bar(vendor_price, title="Top Vendors by Average Unit Price")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No vendor data available")
            
    with tab3:
        if not df.empty:
            df['anomaly'] = detect_anomalies(df)
            fig = px.scatter(
                df, 
                x='Price_Source_Timestamp', 
                y='Unit_Price_Latest', 
                color='anomaly',
                title="Unit Price Anomalies",
                hover_data=['Material_Name', 'Vendor_Name', 'Material_Type']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for anomaly detection")

# def predictive_analytics(df, xgb_model):
#     st.header("🔮 Predictive Pricing Insights")
    
#     tab1, tab2 = st.tabs(["Price Prediction", "Spend Forecasting"])
    
#     with tab1:
#         if xgb_model:
#             st.subheader("Material Price Prediction")
#             with st.form("price_prediction_form"):
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     material_type = st.selectbox(
#                         "Material Type", 
#                         df['Material_Type'].unique(),
#                         key="pred_material_type_select"
#                     )
#                     vendor = st.selectbox(
#                         "Vendor", 
#                         df['Vendor_Name'].unique(),
#                         key="pred_vendor_select"
#                     )
#                     gmp = st.selectbox(
#                         "GMP Compliance", 
#                         ['Yes', 'No'],
#                         key="pred_gmp_select"
#                     )
#                     quantity = st.number_input(
#                         "Quantity Ordered",
#                         min_value=1,
#                         value=1,
#                         key="pred_quantity_input"
#                     )
                    
#                 with col2:
#                     specification = st.selectbox(
#                         "Specification", 
#                         df['Specification'].unique(),
#                         key="pred_spec_select"
#                     )
#                     form = st.selectbox(
#                         "Form", 
#                         df['Form'].unique(),
#                         key="pred_form_select"
#                     )
#                     grade = st.selectbox(
#                         "Grade", 
#                         df['Material_Grade'].unique(),
#                         key="pred_grade_select"
#                     )
#                     order_date = st.date_input(
#                         "Order Date",
#                         value=datetime.now(),
#                         key="pred_date_input"
#                     )
                
#                 submit_button = st.form_submit_button("Predict Price")
                
#                 if submit_button:
#                     try:
#                         # Prepare input data with all required features
#                         input_data = pd.DataFrame([{
#                             'Material_Type': material_type,
#                             'Vendor_Name': vendor,
#                             'GMP_Compliance': gmp,
#                             'Specification': specification,
#                             'Form': form,
#                             'Material_Grade': grade,
#                             'Quantity_Ordered': quantity,
#                             'Year': order_date.year,
#                             'Month': order_date.month
#                         }])
                        
#                         # Encode categorical variables
#                         categorical_cols = ['Material_Type', 'Vendor_Name', 'GMP_Compliance',
#                                           'Specification', 'Form', 'Material_Grade']
#                         for col in categorical_cols:
#                             if col in input_data.columns:
#                                 input_data[col] = input_data[col].astype('category').cat.codes
                        
#                         input_data = input_data[xgb_model.feature_names_in_]
#                         prediction = xgb_model.predict(input_data)[0]
                        
#                         st.success(f"Predicted Unit Price: ${prediction:,.2f}")
                        
#                         # Feature importance plot (without key parameter)
#                         try:
#                             import xgboost
#                             fig, ax = plt.subplots(figsize=(10, 4))
#                             xgboost.plot_importance(xgb_model, ax=ax)
#                             st.pyplot(fig, use_container_width=True)
#                         except Exception as e:
#                             st.warning(f"Couldn't display feature importance: {str(e)}")
                            
#                     except Exception as e:
#                         st.error(f"Prediction failed: {str(e)}")

#         else:
#             st.warning("XGBoost model not available")

#     with tab2:
#         st.subheader("6-Month Spend Forecast")
        
#         if 'Price_Source_Timestamp' not in df.columns or 'PO_Amount' not in df.columns:
#             st.error("Required columns missing")
#         elif df['Price_Source_Timestamp'].isnull().any():
#             st.error("Missing dates in data")
#         else:
#             forecast = forecast_spend(df)
            
#             if forecast is not None:
#                 fig = px.line(
#                     forecast, 
#                     x='ds', 
#                     y='yhat',
#                     title='6-Month Spend Forecast',
#                     labels={'ds': 'Date', 'yhat': 'Amount'}
#                 )
#                 st.plotly_chart(
#                     fig, 
#                     use_container_width=True,
#                     key="spend_forecast_chart"
#                 )
                
#                 # Expander without key parameter
#                 with st.expander("Forecast Details"):
#                     st.dataframe(
#                         forecast.style.format({
#                             'yhat': '${:,.2f}',
#                             'yhat_lower': '${:,.2f}', 
#                             'yhat_upper': '${:,.2f}'
#                         }),
#                         key="forecast_details_df"
#                     )
#             else:
#                 st.warning("Could not generate forecast. Check data requirements.")


def predictive_analytics(df, xgb_model):
    st.header("🔮 Predictive Pricing Insights")
    
    # Create tabs for different predictive functions
    tab1, tab2 = st.tabs(["📈 Price Prediction", "📊 Spend Forecasting"])
    
    with tab1:
        st.subheader("Material Price Prediction")
        
        if not xgb_model:
            st.warning("Price prediction model not available")
            return
        
        # Create a form for all inputs
        with st.form("price_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Get unique values safely with defaults
                material_types = df['Material_Type'].unique() if 'Material_Type' in df.columns else ['Unknown']
                material_type = st.selectbox(
                    "Material Type", 
                    material_types,
                    index=0 if len(material_types) > 0 else None
                )
                
                vendors = df['Vendor_Name'].unique() if 'Vendor_Name' in df.columns else ['Unknown']
                vendor = st.selectbox(
                    "Vendor", 
                    vendors,
                    index=0 if len(vendors) > 0 else None
                )
                
                gmp = st.selectbox(
                    "GMP Compliance", 
                    ['Yes', 'No'],
                    index=0
                )
                
                quantity = st.number_input(
                    "Quantity Ordered",
                    min_value=1,
                    value=100,
                    step=10
                )
                
            with col2:
                specs = df['Specification'].unique() if 'Specification' in df.columns else ['Unknown']
                specification = st.selectbox(
                    "Specification", 
                    specs,
                    index=0 if len(specs) > 0 else None
                )
                
                forms = df['Form'].unique() if 'Form' in df.columns else ['Unknown']
                form = st.selectbox(
                    "Form", 
                    forms,
                    index=0 if len(forms) > 0 else None
                )
                
                grades = df['Material_Grade'].unique() if 'Material_Grade' in df.columns else ['Unknown']
                grade = st.selectbox(
                    "Grade", 
                    grades,
                    index=0 if len(grades) > 0 else None
                )
                
                order_date = st.date_input(
                    "Order Date",
                    value=datetime.now()
                )
            
            submit_button = st.form_submit_button("Predict Price")
        
        if submit_button:
            with st.spinner("Generating price prediction..."):
                try:
                    # Prepare input data with all required features
                    input_data = pd.DataFrame([{
                        'Material_Type': material_type,
                        'Vendor_Name': vendor,
                        'GMP_Compliance': 1 if gmp == 'Yes' else 0,
                        'Specification': specification,
                        'Form': form,
                        'Material_Grade': grade,
                        'Quantity_Ordered': quantity,
                        'price_year': order_date.year,
                        'price_month': order_date.month
                    }])
                    
                    # Ensure we only use features the model expects
                    model_features = xgb_model.get_booster().feature_names
                    missing_features = [f for f in model_features if f not in input_data.columns]
                    
                    if missing_features:
                        st.warning(f"Adding default values for missing features: {', '.join(missing_features)}")
                        for f in missing_features:
                            input_data[f] = 0  # Or appropriate default value
                    
                    # Keep only the features the model expects
                    input_data = input_data[model_features]
                    
                    # Make prediction
                    prediction = xgb_model.predict(input_data)[0]
                    
                    # Display results
                    st.success("### Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Unit Price", f"${prediction:,.2f}")
                    
                    with col2:
                        # Find similar historical prices for context
                        similar_prices = df[
                            (df['Material_Type'] == material_type) & 
                            (df['Vendor_Name'] == vendor)
                        ]['Unit_Price_Latest']
                        
                        if not similar_prices.empty:
                            avg_price = similar_prices.mean()
                            diff = prediction - avg_price
                            pct_diff = (diff / avg_price) * 100 if avg_price != 0 else 0
                            st.metric(
                                "Compared to Historical Average",
                                f"${avg_price:,.2f}",
                                delta=f"{pct_diff:.1f}% {'higher' if diff > 0 else 'lower'}",
                                delta_color="inverse" if abs(pct_diff) > 10 else "normal"
                            )
                    
                    # Feature importance visualization
                    st.subheader("Key Factors Influencing This Prediction")
                    try:
                        import xgboost
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        xgboost.plot_importance(
                            xgb_model,
                            ax=ax,
                            importance_type='weight',
                            max_num_features=10,
                            height=0.8
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Couldn't display feature importance: {str(e)}")
                    
                    # Show similar historical transactions
                    st.subheader("Similar Historical Transactions")
                    similar_transactions = df[
                        (df['Material_Type'] == material_type) &
                        (df['Vendor_Name'] == vendor)
                    ].sort_values('Price_Source_Timestamp', ascending=False)
                    
                    if not similar_transactions.empty:
                        st.dataframe(
                            similar_transactions[[
                                'Material_Name',
                                'Unit_Price_Latest',
                                'Quantity_Ordered',
                                'Price_Source_Timestamp',
                                'Price_Deviation (%)'
                            ]].head(5).style.format({
                                'Unit_Price_Latest': '${:,.2f}',
                                'Price_Deviation (%)': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("No similar historical transactions found")
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.error("Please check that all required input fields are available in your data.")
    with tab2:
        st.subheader("6-Month Spend Forecast")
        
        if 'Price_Source_Timestamp' not in df.columns or 'PO_Amount' not in df.columns:
            st.error("Required columns missing for forecasting")
        else:
            with st.spinner("Generating spend forecast..."):
                try:
                    # Prepare time series data
                    ts = df.resample('M', on='Price_Source_Timestamp')['PO_Amount']\
                        .sum()\
                        .reset_index()
                    ts.columns = ['ds', 'y']
                    
                    # Calculate available months
                    available_months = len(ts)
                    min_required_months = 3  # Reduced from 6 to make it work with less data
                    
                    if available_months < min_required_months:
                        # st.warning(f"""
                        # Insufficient historical data ({available_months} month{'s' if available_months != 1 else ''} available).
                        # At least {min_required_months} months recommended for forecasting.
                        # """)
                        
                        # Show available historical data
                        st.subheader("Available Historical Data")
                        st.line_chart(ts.set_index('ds'))
                        return
                    
                    # Adjust forecast period based on available data
                    forecast_period = min(6, available_months)  # Don't forecast more months than we have history
                    
                    # Train model
                    model = Prophet(
                        yearly_seasonality=available_months >= 12,  # Only enable yearly seasonality if we have 1+ year
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.5  # More conservative with less data
                    )
                    model.fit(ts)
                    
                    # Create future dataframe
                    future = model.make_future_dataframe(periods=forecast_period, freq='M')
                    
                    # Generate forecast
                    forecast = model.predict(future)
                    
                    # Display results
                    st.success(f"{forecast_period}-Month Spend Forecast")
                    
                    # Plot forecast with confidence intervals
                    fig1 = model.plot(forecast)
                    plt.title(f'{forecast_period}-Month Spend Forecast')
                    plt.xlabel('Date')
                    plt.ylabel('Amount ($)')
                    
                    # Highlight actual vs forecast
                    plt.scatter(ts['ds'], ts['y'], color='red', label='Actual', zorder=10)
                    plt.legend()
                    st.pyplot(fig1)
                    
                    # Show forecast details
                    with st.expander("Forecast Details", expanded=False):
                        st.dataframe(
                            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]\
                                .tail(forecast_period + 1)\
                                .style.format({
                                    'yhat': '${:,.2f}',
                                    'yhat_lower': '${:,.2f}',
                                    'yhat_upper': '${:,.2f}'
                                }),
                            use_container_width=True
                        )
                    
                    # Add disclaimer for limited data
                    if available_months < 6:
                        st.warning("""
                        **Note:** Forecast reliability is reduced with limited historical data. 
                        Consider collecting more data for more accurate predictions.
                        """)
                
                except Exception as e:
                    st.error(f"Forecasting failed: {str(e)}")


def vendor_intelligence(df):
    st.header("🏆 Vendor Performance Hub")
    
    st.subheader("Vendor-Material Relationship Matrix")
    
    if 'Vendor_Name' not in df.columns or 'Material_Name' not in df.columns:
        st.warning("Cannot display relationships - missing vendor or material data")
        return
    
    vendor_material = df.groupby(['Vendor_Name', 'Material_Name']).size().unstack().fillna(0)
    
    if not vendor_material.empty:
        st.dataframe(
            vendor_material.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        st.markdown("""
        ### 🔍 Understanding the Relationship Matrix
        
        This table shows how many pricing records exist between each vendor (rows) and material (columns):
        
        - **Vendor**: Pharmaceutical supplier (listed on the left)
        - **Material**: Chemical or ingredient (listed on top)
        - **Cell Value**: Number of price records for that vendor-material combination
        """)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("**Unique Vendors**", 
                      vendor_material.shape[0],
                      help="Total number of different suppliers in our system")
        
        with col2:
            st.metric("**Unique Materials**", 
                      vendor_material.shape[1],
                      help="Different pharmaceutical materials being purchased")
        
        with col3:
            st.metric("**Active Relationships**", 
                      (vendor_material > 0).sum().sum(),
                      help="Vendor-Material pairs with at least one price record")
        
        st.markdown("""
        ### 💡 What This Tells Us:
        
        1. **Vendor Specialization**: 
           - Some vendors may specialize in certain material types
           - Others may offer a broad range of materials
        
        2. **Material Sourcing**:
           - Some materials may have multiple vendor options
           - Others may have limited supplier options
        
        3. **Opportunities**:
           - Potential to consolidate vendors where possible
           - Chance to negotiate better terms with key suppliers
        """)
        
    else:
        st.warning("No vendor-material relationships found in the data")
    
    st.subheader("Vendor Performance Metrics")
    perf_df = vendor_performance(df)
    if not perf_df.empty:
        st.dataframe(
            perf_df.style.format({
                'Total Spend': '${:,.2f}',
                'Avg PO Value': '${:,.2f}'
            }).background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        st.markdown("""
        ### 📊 Understanding Performance Metrics:
        
        - **PO Count**: How many purchase orders with this vendor
        - **Total Spend**: Combined value of all orders
        - **Avg PO Value**: Average size of each order
        - **Days Since Last PO**: How recently we worked with them
        """)
    else:
        st.warning("No vendor performance data available")

def spend_optimizer(df):
    st.header("💵 Pharma Savings Opportunity Engine")
    
    tab1, tab2 = st.tabs(["Price Benchmarking", "Spend Consolidation"])
    
    with tab1:
        st.subheader("Price Benchmarking Analysis")
        if not df.empty:
            benchmark_threshold = st.number_input("Price Deviation Threshold (%)", 
                                              value=10.0,
                                              min_value=0.0,
                                              max_value=100.0,
                                              key="benchmark_thresh")
            
            high_deviation = df[df['Price_Deviation (%)'].abs() > benchmark_threshold]
            total = len(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("High Deviation Records", 
                         f"{len(high_deviation)} ({len(high_deviation)/total:.1%})",
                         delta=f"Threshold: {benchmark_threshold}%")
            
            with col2:
                st.metric("Potential Savings", 
                         f"${(high_deviation['PO_Amount'].sum() * (high_deviation['Price_Deviation (%)']/100)).sum():,.2f}",
                         delta="If deviations were eliminated")
            
            if not high_deviation.empty:
                with st.expander("View High Deviation Records", expanded=False):
                    st.dataframe(
                        high_deviation[['Material_Name', 'Vendor_Name', 'Price_Deviation (%)', 'PO_Amount']]
                        .sort_values('Price_Deviation (%)', ascending=False),
                        use_container_width=True
                    )
        else:
            st.warning("No data available for benchmarking analysis")
    
    with tab2:
        st.subheader("Spend Consolidation Opportunities")
        if 'Vendor_Name' in df.columns and 'PO_Amount' in df.columns:
            vendor_spend = df.groupby('Vendor_Name')['PO_Amount'].agg(['count', 'sum'])
            vendor_spend = vendor_spend.sort_values('sum', ascending=False)
            
            st.write("**Top Vendors by Spend**")
            st.dataframe(
                vendor_spend.head(10).style.format({
                    'count': '{:,}',
                    'sum': '${:,.2f}'
                }),
                use_container_width=True
            )
            
            small_orders = vendor_spend[(vendor_spend['count'] > 1) & 
                                       (vendor_spend['sum'] < vendor_spend['sum'].quantile(0.75))]
            
            if not small_orders.empty:
                st.write("**Potential Consolidation Candidates**")
                st.write("Vendors with multiple small orders:")
                st.dataframe(
                    small_orders.style.format({
                        'count': '{:,}',
                        'sum': '${:,.2f}'
                    }),
                    use_container_width=True
                )
        else:
            st.warning("Vendor or spend data not available for consolidation analysis")

def procurement_recommendation_agent(df):
    st.header("🤖 AI Procurement Recommendation Engine")
    
    tab1, tab2, tab3 = st.tabs(["📊 Smart Recommendations", "💬 Negotiation Simulator", "📈 Strategy Analytics"])
    
    with tab1:
        st.subheader("Intelligent Procurement Recommendations")
        
        if df.empty:
            st.warning("No data available for recommendations")
            return
        
        # Material selection
        selected_material = st.selectbox(
            "Select Material for Recommendations",
            df['Material_Name'].unique(),
            index=0 if len(df['Material_Name'].unique()) > 0 else None
        )
        
        if not selected_material:
            st.warning("Please select a material")
            return
            
        material_data = df[df['Material_Name'] == selected_material]
        
        if material_data.empty:
            st.warning(f"No data found for material: {selected_material}")
            return
        
        with st.spinner("Generating AI recommendations..."):
            try:
                # Analyze price trends
                price_trend = material_data.groupby(
                    material_data['Price_Source_Timestamp'].dt.to_period('M')
                )['Unit_Price_Latest'].mean().reset_index()
                price_trend['Price_Source_Timestamp'] = price_trend['Price_Source_Timestamp'].astype(str)
                
                # Vendor analysis
                vendor_stats = material_data.groupby('Vendor_Name').agg({
                    'Unit_Price_Latest': ['mean', 'std', 'count'],
                    'PO_Amount': 'sum'
                }).reset_index()
                vendor_stats.columns = ['Vendor', 'Avg Price', 'Price Std Dev', 'Order Count', 'Total Spend']
                
                # Generate AI recommendations
                recommendation_prompt = f"""
                Analyze this pharmaceutical procurement data and provide recommendations:
                
                Material: {selected_material}
                
                Price Trends:
                {price_trend.to_string(index=False)}
                
                Vendor Performance:
                {vendor_stats.to_string(index=False)}
                
                Provide:
                1. Best time to buy (based on historical price patterns)
                2. Recommended vendors (considering price, reliability, and volume)
                3. Optimal order quantity strategy
                4. Risk factors to consider
                """
                
                openai.api_key = OPENAI_API_KEY
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a pharmaceutical procurement expert."},
                        {"role": "user", "content": recommendation_prompt}
                    ],
                    temperature=0.7
                )
                
                recommendations = response.choices[0].message.content
                
                # Display results
                st.subheader(f"Recommendations for {selected_material}")
                st.markdown(recommendations)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Price Trend")
                    fig = px.line(
                        price_trend,
                        x='Price_Source_Timestamp',
                        y='Unit_Price_Latest',
                        title=f"{selected_material} Monthly Price Trend"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Vendor Comparison")
                    fig = px.bar(
                        vendor_stats.sort_values('Avg Price'),
                        x='Vendor',
                        y='Avg Price',
                        error_y='Price Std Dev',
                        title="Vendor Price Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Recommendation generation failed: {str(e)}")
    
    with tab2:
        st.subheader("AI Negotiation Simulator")
        
        # Select negotiation scenario
        scenario = st.selectbox(
            "Select Negotiation Scenario",
            [
                "Price Reduction Request",
                "Contract Term Negotiation",
                "Volume Discount Discussion",
                "Quality Dispute Resolution"
            ]
        )
        
        # Set up negotiation parameters
        with st.form("negotiation_params"):
            col1, col2 = st.columns(2)
            with col1:
                vendor = st.selectbox(
                    "Vendor",
                    df['Vendor_Name'].unique(),
                    index=0 if len(df['Vendor_Name'].unique()) > 0 else None
                )
                current_price = st.number_input(
                    "Current Unit Price",
                    value=float(df[df['Vendor_Name']==vendor]['Unit_Price_Latest'].mean()) if vendor else 100.0,
                    min_value=0.01
                )
            with col2:
                material = st.selectbox(
                    "Material",
                    df[df['Vendor_Name']==vendor]['Material_Name'].unique() if vendor else df['Material_Name'].unique(),
                    index=0
                )
                target_price = st.number_input(
                    "Your Target Price",
                    value=current_price * 0.9,  # Default to 10% reduction
                    min_value=0.01
                )
            
            negotiation_strategy = st.text_area(
                "Your Negotiation Strategy",
                value=f"We would like to discuss a price reduction for {material} from ${current_price:.2f} to ${target_price:.2f} based on market benchmarks."
            )
            
            submitted = st.form_submit_button("Start Negotiation Simulation")
        
        if submitted:
            with st.spinner("Simulating negotiation..."):
                try:
                    # Get vendor historical data
                    vendor_history = df[(df['Vendor_Name'] == vendor) & 
                                      (df['Material_Name'] == material)]
                    
                    # Prepare negotiation prompt
                    negotiation_prompt = f"""
                    Analyze this negotiation scenario and provide structured insights:
                    
                    Scenario: {scenario}
                    Vendor: {vendor}
                    Material: {material}
                    Current Price: ${current_price:.2f}
                    Target Price: ${target_price:.2f}
                    Strategy: "{negotiation_strategy}"
                    
                    Vendor History:
                    - Avg Price: ${vendor_history['Unit_Price_Latest'].mean():.2f}
                    - Min Price: ${vendor_history['Unit_Price_Latest'].min():.2f}
                    - Max Price: ${vendor_history['Unit_Price_Latest'].max():.2f}
                    - Order Count: {len(vendor_history)}
                    - Total Spend: ${vendor_history['PO_Amount'].sum():,.2f}
                    
                    Provide analysis in this exact JSON format:
                    {{
                        "vendor_response": "concise vendor reply (1-2 sentences)",
                        "likely_outcomes": [
                            {{"scenario": "Best Case", "probability": "X%", "price": "$X.XX", "terms": "key terms"}},
                            {{"scenario": "Most Likely", "probability": "X%", "price": "$X.XX", "terms": "key terms"}},
                            {{"scenario": "Worst Case", "probability": "X%", "price": "$X.XX", "terms": "key terms"}}
                        ],
                        "compromise_options": [
                            {{"option": "description", "price_impact": "+/-X%", "value": "benefit"}},
                            {{"option": "description", "price_impact": "+/-X%", "value": "benefit"}}
                        ],
                        "next_steps": ["step1", "step2", "step3"]
                    }}
                    """
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert pharmaceutical procurement negotiator. Provide structured, data-driven negotiation analysis in exact specified JSON format."},
                            {"role": "user", "content": negotiation_prompt}
                        ],
                        temperature=0.7
                    )
                    
                    # Parse the JSON response
                    negotiation_data = json.loads(response.choices[0].message.content)
                    
                    # Display results in visual format
                    st.subheader("🎯 Negotiation Simulation Results")
                    
                    # Vendor Response Card
                    with st.expander("💬 Vendor's Likely Response", expanded=True):
                        st.info(negotiation_data["vendor_response"])
                    
                    # Outcome Probability Visualization
                    st.subheader("📊 Expected Outcomes")
                    outcomes_df = pd.DataFrame(negotiation_data["likely_outcomes"])
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Target Price", f"${target_price:.2f}")
                    
                    with col2:
                        fig = px.bar(
                            outcomes_df,
                            x='scenario',
                            y='probability',
                            color='scenario',
                            text='price',
                            title="Negotiation Outcome Probabilities"
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Compromise Options
                    st.subheader("🤝 Potential Compromise Options")
                    compromises = pd.DataFrame(negotiation_data["compromise_options"])
                    
                    for idx, row in compromises.iterrows():
                        with st.container():
                            cols = st.columns([1, 4, 2])
                            cols[0].metric("Impact", row["price_impact"])
                            cols[1].write(f"**{row['option']}**")
                            cols[1].write(row["value"])
                            if cols[2].button("Explore", key=f"comp_{idx}"):
                                st.session_state[f"comp_detail_{idx}"] = not st.session_state.get(f"comp_detail_{idx}", False)
                            
                            if st.session_state.get(f"comp_detail_{idx}", False):
                                st.info(f"Detailed analysis for: {row['option']}")
                                # Here you could add more detailed analysis if needed
                    
                    # Next Steps
                    st.subheader("🚀 Recommended Next Steps")
                    for i, step in enumerate(negotiation_data["next_steps"]):
                        st.checkbox(f"{i+1}. {step}", value=False, key=f"step_{i}")
                    
                    # Market Context Visualization
                    st.subheader("📈 Market Context")
                    
                    # Create comparison with other vendors
                    market_comparison = df[df['Material_Name'] == material].groupby('Vendor_Name').agg({
                        'Unit_Price_Latest': ['mean', 'count'],
                        'PO_Amount': 'sum'
                    }).reset_index()
                    market_comparison.columns = ['Vendor', 'Avg Price', 'Order Count', 'Total Spend']
                    
                    fig = px.scatter(
                        market_comparison,
                        x='Avg Price',
                        y='Order Count',
                        size='Total Spend',
                        color='Vendor',
                        hover_name='Vendor',
                        title=f"Market Comparison for {material}",
                        labels={'Avg Price': 'Average Price ($)', 'Order Count': 'Number of Orders'}
                    )
                    fig.add_vline(x=current_price, line_dash="dash", line_color="red", annotation_text="Current Price")
                    fig.add_vline(x=target_price, line_dash="dash", line_color="green", annotation_text="Target Price")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Historical Price Trend
                    if not vendor_history.empty and 'Price_Source_Timestamp' in vendor_history.columns:
                        st.subheader("🕒 Historical Price Trend")
                        history_df = vendor_history.sort_values('Price_Source_Timestamp')
                        fig = px.line(
                            history_df,
                            x='Price_Source_Timestamp',
                            y='Unit_Price_Latest',
                            title=f"{vendor}'s Price History for {material}",
                            markers=True
                        )
                        fig.add_hline(y=current_price, line_dash="dash", line_color="red", annotation_text="Current")
                        fig.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text="Target")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # BATNA Analysis
                    with st.expander("🔍 BATNA (Best Alternative)", expanded=False):
                        alternatives = df[
                            (df['Material_Name'] == material) & 
                            (df['Vendor_Name'] != vendor)
                        ].groupby('Vendor_Name')['Unit_Price_Latest'].mean().reset_index()
                        
                        if not alternatives.empty:
                            st.write("**Alternative Vendors**")
                            fig = px.bar(
                                alternatives.sort_values('Unit_Price_Latest'),
                                x='Vendor_Name',
                                y='Unit_Price_Latest',
                                title="Alternative Vendor Prices",
                                labels={'Unit_Price_Latest': 'Price ($)', 'Vendor_Name': 'Vendor'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            best_alt = alternatives['Unit_Price_Latest'].min()
                            st.metric("Best Alternative Price", f"${best_alt:.2f}", 
                                     delta=f"${current_price-best_alt:.2f} vs current" if current_price > best_alt else "")
                        else:
                            st.warning("No alternative vendors found for this material")
                
                except Exception as e:
                    st.error(f"Negotiation simulation failed: {str(e)}")
    
    with tab3:
        st.subheader("Procurement Strategy Analytics")
        
        if df.empty:
            st.warning("No data available for analysis")
            return
        
        # Strategy selection
        strategy = st.selectbox(
            "Select Analysis Type",
            [
                "Vendor Consolidation Opportunities",
                "Price Benchmarking Gaps",
                "Seasonal Buying Patterns",
                "Risk Concentration Analysis"
            ]
        )
        
        if st.button("Run Strategic Analysis"):
            with st.spinner("Analyzing procurement strategy..."):
                try:
                    # Prepare analysis based on selected strategy
                    if strategy == "Vendor Consolidation Opportunities":
                        vendor_summary = df.groupby('Vendor_Name').agg({
                            'Material_Name': 'nunique',
                            'PO_Amount': ['count', 'sum']
                        }).reset_index()
                        vendor_summary.columns = ['Vendor', 'Unique Materials', 'Order Count', 'Total Spend']
                        
                        st.write("**Vendor Spend Concentration**")
                        fig = px.treemap(
                            vendor_summary,
                            path=['Vendor'],
                            values='Total Spend',
                            color='Unique Materials',
                            title="Vendor Spend Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Identify consolidation candidates
                        small_vendors = vendor_summary[
                            (vendor_summary['Total Spend'] < vendor_summary['Total Spend'].quantile(0.5)) &
                            (vendor_summary['Order Count'] > 1)
                        ]
                        
                        if not small_vendors.empty:
                            st.subheader("Vendor Consolidation Candidates")
                            st.write("These vendors have multiple small orders that could potentially be consolidated:")
                            st.dataframe(
                                small_vendors.sort_values('Order Count', ascending=False),
                                use_container_width=True
                            )
                    
                    elif strategy == "Price Benchmarking Gaps":
                        price_comparison = df.groupby('Material_Name').agg({
                            'Unit_Price_Latest': ['mean', 'std'],
                            'Benchmark_Price': 'mean',
                            'Vendor_Name': 'nunique'
                        }).reset_index()
                        price_comparison.columns = [
                            'Material', 'Avg Price', 'Price Std Dev', 
                            'Benchmark Price', 'Vendor Count'
                        ]
                        
                        price_comparison['Price Variance'] = (
                            (price_comparison['Avg Price'] - price_comparison['Benchmark Price']) / 
                            price_comparison['Benchmark Price']
                        ) * 100
                        
                        st.write("**Materials with Largest Price Variance from Benchmark**")
                        fig = px.bar(
                            price_comparison.nlargest(10, 'Price Variance'),
                            x='Material',
                            y='Price Variance',
                            title="Top Materials by Price Variance from Benchmark",
                            hover_data=['Avg Price', 'Benchmark Price', 'Vendor Count']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif strategy == "Seasonal Buying Patterns":
                        if 'Price_Source_Timestamp' not in df.columns:
                            st.warning("Date information missing for seasonal analysis")
                            return
                            
                        monthly_trends = df.groupby(
                            df['Price_Source_Timestamp'].dt.month
                        )['PO_Amount'].sum().reset_index()
                        
                        fig = px.line(
                            monthly_trends,
                            x='Price_Source_Timestamp',
                            y='PO_Amount',
                            title="Monthly Procurement Spend Patterns"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif strategy == "Risk Concentration Analysis":
                        risk_metrics = df.groupby('Material_Name').agg({
                            'Vendor_Name': 'nunique',
                            'PO_Amount': 'sum'
                        }).reset_index()
                        risk_metrics.columns = ['Material', 'Vendor Count', 'Total Spend']
                        
                        fig = px.scatter(
                            risk_metrics,
                            x='Vendor Count',
                            y='Total Spend',
                            size='Total Spend',
                            color='Vendor Count',
                            hover_name='Material',
                            title="Supply Risk Analysis (Few Vendors + High Spend = High Risk)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        high_risk = risk_metrics[
                            (risk_metrics['Vendor Count'] <= 2) & 
                            (risk_metrics['Total Spend'] > risk_metrics['Total Spend'].median())
                        ]
                        
                        if not high_risk.empty:
                            st.subheader("High-Risk Materials")
                            st.write("These materials have limited vendor options and significant spend:")
                            st.dataframe(high_risk, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Strategic analysis failed: {str(e)}")

def procurement_assistant(df):
    st.header("💊 Pharma Procurement Assistant")
    
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    df.to_sql('pharma_data', conn, index=False, if_exists='replace')

    def generate_sql_query(question: str) -> str:
        system_prompt = """
        You are a SQL expert analyzing pharmaceutical procurement data. Follow these rules:
        
        1. DATABASE SCHEMA:
        - Columns: Material_Name, Material_Type, Vendor_Name, GMP_Compliance, Specification, 
          Form, Material_Grade, Unit_Price_Latest, Benchmark_Price, [Price_Deviation (%)],
          Quantity_Ordered, PO_Amount, Price_Source_Timestamp, Internal_Contract_Price,
          Portal_Price, [Contract_vs_Latest (%)]

        2. REQUIREMENTS:
        - Always use SQLite syntax
        - For percentage columns (like [Price_Deviation (%)]), use ABS() when comparing
        - Include Material_Name and Vendor_Name in results for context
        - Sort by most relevant metric (e.g., highest deviation first)
        - Never use LIMIT unless explicitly requested
        - Handle special characters with square brackets []
        - Return ONLY the SQL query, no explanations

        3. EXAMPLES:
        Good: "SELECT Material_Name, [Price_Deviation (%)] FROM pharma_data WHERE ABS([Price_Deviation (%)]) > 20"
        Bad: "Here's your query: SELECT..."
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                api_key=OPENAI_API_KEY,
                temperature=0.3  # More deterministic output
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Query generation failed: {str(e)}")
            return None

    def execute_query_and_analyze(conn: Connection, sql_query: str, original_question: str) -> tuple:
        try:
            result_df = pd.read_sql(sql_query, conn)
            
            if result_df.empty:
                # Suggest potential fixes for empty results
                analysis = (
                    "No results found. Possible reasons:\n"
                    "1. Your criteria may be too strict (try broadening filters)\n"
                    "2. The requested data may not exist in our records\n"
                    "3. Column names may differ from expected\n\n"
                    "Try questions like:\n"
                    "- 'Show 5 sample materials with their prices'\n"
                    "- 'What columns are available for analysis?'"
                )
                return analysis, None
            
            # Enhanced analysis
            analysis_prompt = f"""
            Perform pharmaceutical procurement analysis with these rules:
            
            1. CONTEXT: User asked: "{original_question}"
            2. DATA: {result_df.to_string(index=False, max_rows=10)}
            3. TASKS:
               - Identify key insights (top items, outliers, trends)
               - Note data limitations (sample size, missing values)
               - Suggest follow-up questions
            4. FORMAT:
               - Start with direct answer to the question
               - Then list 2-3 key observations
               - End with 1-2 suggested next analyses
            
            Example:
            "The top 3 materials by price deviation are X, Y, Z. 
            Key observations:
            1. Vendor A consistently has higher deviations
            2. 60% of deviations are for injectable forms
            Consider analyzing: 'Show price trends for Vendor A's injectables'"
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a pharmaceutical procurement analyst."},
                    {"role": "user", "content": analysis_prompt}
                ],
                api_key=OPENAI_API_KEY
            )
            return response.choices[0].message.content.strip(), result_df

        except Exception as e:
            return f"❌ Error: {str(e)}\n\nTry simplifying your question or checking column names.", None

    # UI Enhancements
    with st.expander("💡 Example Questions", expanded=True):
        st.markdown("""
        - **Price Analysis**: "Show materials with >20% price deviation from benchmark"
        - **Vendor Performance**: "List top 5 vendors by total spend"
        - **Savings Opportunities**: "Find contracts where portal price is cheaper than contract price"
        - **Data Exploration**: "What materials have the highest inventory vs latest price differences?"
        """)

    question = st.text_input("🔍 Ask a question about the pharmaceutical pricing dataset:",
                           placeholder="e.g., 'Show materials with price deviations >20%'")

    if question:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Generating optimized SQL query..."):
                sql_query = generate_sql_query(question)
                if sql_query:
                    st.subheader("📝 Generated SQL Query")
                    st.code(sql_query, language="sql")
                
        with col2:
            if sql_query:
                with st.spinner("Executing and analyzing..."):
                    analysis, results_df = execute_query_and_analyze(conn, sql_query, question)
                
                if results_df is not None:
                    st.subheader(f"📊 Results ({len(results_df)} records)")
                    st.dataframe(results_df.style.format({
                        'Unit_Price_Latest': '{:.2f}',
                        'PO_Amount': '{:,.2f}',
                        'Price_Deviation (%)': '{:.1f}%'
                    }), height=400)

        if sql_query and analysis:
            st.subheader("🧠 AI Analysis")
            st.markdown(analysis)

    # Data Documentation Sidebar
    with st.sidebar:
        st.title("📁 Data Guide")
        st.metric("Total Records", len(df))
        st.metric("Unique Materials", df['Material_Name'].nunique())
        
        with st.expander("Key Columns"):
            st.markdown("""
            - **Material_Name**: Name of pharmaceutical product
            - **Vendor_Name**: Supplier name  
            - **Unit_Price_Latest**: Current price
            - **Benchmark_Price**: Reference price
            - **[Price_Deviation (%)]**: % difference from benchmark
            - **PO_Amount**: Total order value
            """)
        
        if st.checkbox("Show raw data sample"):
            st.dataframe(df.head())

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket

# # Configuration (move to secrets in production)
# SMTP_CONFIG = {
#     "server": "smtp.gmail.com",  # Replace with your SMTP server
#     "port": 587,
#     "sender": "procurement.alerts@yourdomain.com",
#     "password": st.secrets["smtp_password"],  # Set in Streamlit secrets
#     "recipients": ["procurement-team@yourdomain.com", "manager@yourdomain.com"]
# }
SMTP_CONFIG = {
    "server": "smtp.gmail.com",
    "port": 587,
    "sender": "priyangshu@elementtechnologies.com",
    # Use get() with a default value if secret isn't available
    "password": st.secrets.get("smtp_password", "dummy_password"),
    "recipients": ["priyangshusarkar07@gmail.com"]
}
def send_email_alert(subject, message, recipients=None, priority="high"):
    """Sends email alerts for procurement decisions"""
    try:
        if recipients is None:
            recipients = SMTP_CONFIG["recipients"]
            
        msg = MIMEMultipart()
        msg['From'] = SMTP_CONFIG["sender"]
        msg['To'] = ", ".join(recipients)
        urgent_prefix = "[URGENT] " if priority == "high" else ""
        msg['Subject'] = f"{urgent_prefix}{subject}"
        
        # Fix 1: Store backslash characters in variables
        newline = "\n"
        html_message = message.replace(newline, '<br>')
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        html = f"""
        <html>
            <body>
                <h2>Pharma Procurement Alert</h2>
                <p><strong>Priority:</strong> {priority.upper()}</p>
                <div style="background-color:#f8f9fa;padding:15px;border-radius:5px;">
                    {html_message}
                </div>
                <p><em>Generated by Pharma Procurement AI at {current_time}</em></p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        with smtplib.SMTP(SMTP_CONFIG["server"], SMTP_CONFIG["port"]) as server:
            server.starttls()
            server.login(SMTP_CONFIG["sender"], SMTP_CONFIG["password"])
            server.sendmail(SMTP_CONFIG["sender"], recipients, msg.as_string())
            
        return True
    except Exception as e:
        st.error(f"Failed to send email alert: {str(e)}")
        return False

def generate_alert_content(df, alert_type):
    """Generates human-readable alert content"""
    if alert_type == "exception":
        critical_exceptions = df[df['Exception_Type'] == "Price Exception"]
        if not critical_exceptions.empty:
            alert_df = critical_exceptions.nlargest(5, 'PO_Amount')
            message = [
                "🚨 Critical Price Exceptions Detected:",
                f"Total exceptions: {len(critical_exceptions)}",
                f"Total value at risk: ${critical_exceptions['PO_Amount'].sum():,.2f}",
                "",
                "Top 5 exceptions by value:"
            ]
            
            for _, row in alert_df.iterrows():
                message.append(
                    f"- {row['Material_Name']} from {row['Vendor_Name']}: "
                    f"${row['Unit_Price_Latest']:,.2f} (Deviation: {abs(row['Price_Deviation (%)']):.1f}%) "
                    f"| PO Amount: ${row['PO_Amount']:,.2f}"
                )
            
            return "\n".join(message)
    
    elif alert_type == "approval":
        pending_reviews = df[df['Final_Decision'] == "Review Needed"]
        if not pending_reviews.empty:
            message = [
                "⚠️ Pending Purchase Approvals:",
                f"Total items needing review: {len(pending_reviews)}",
                f"Total value pending approval: ${pending_reviews['PO_Amount'].sum():,.2f}",
                "",
                "Top items requiring attention:"
            ]
            
            for _, row in pending_reviews.nlargest(3, 'PO_Amount').iterrows():
                message.append(
                    f"- {row['Material_Name']} from {row['Vendor_Name']}: "
                    f"${row['Unit_Price_Latest']:,.2f} | Qty: {row['Quantity_Ordered']} "
                    f"| Reason: {row['Approval_Notes']}"
                )
            
            return "\n".join(message)
    
    return None

def send_email_alert(recipients, subject, message):
    import smtplib
    from email.mime.text import MIMEText
    
    # Configure your SMTP settings
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "priyangshusarkar07@gmail.com"
    sender_password = "yawj ryrc okcd vwke"
    
    # Create message
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipients)
    
    # Send email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def decision_layer(df, predictions=None):
    st.header("📋 Procurement Decision Engine")
    
    tab1, tab2, tab3 = st.tabs(["Approval Workflow", "Order Optimization", "Exception Handling"])
    
    with tab1:
        st.subheader("Purchase Approval Recommendations")
        
        # Create approval rules engine
        with st.expander("⚙️ Approval Rules Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                price_threshold = st.number_input(
                    "Price Deviation Threshold (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=15.0
                )
                qty_threshold = st.number_input(
                    "Large Order Threshold", 
                    min_value=0, 
                    value=1000
                )
            with col2:
                vendor_rating_threshold = st.number_input(
                    "Minimum Vendor Rating", 
                    min_value=0.0, 
                    max_value=5.0, 
                    value=3.5,
                    step=0.1
                )
                new_vendor_flag = st.checkbox(
                    "Flag New Vendor Orders", 
                    value=True
                )
        
        if st.button("Run Approval Analysis"):
            with st.spinner("Analyzing purchase requests..."):
                try:
                    # FIXED: Use consistent status values
                    df['Final_Decision'] = "APPROVED"
                    df['Approval_Notes'] = ""
                    
                    # Rule 1: Price deviation
                    if 'Price_Deviation (%)' in df.columns:
                        df.loc[df['Price_Deviation (%)'].abs() > price_threshold, 'Final_Decision'] = "PENDING_REVIEW"
                        df.loc[df['Price_Deviation (%)'].abs() > price_threshold, 'Approval_Notes'] += "High price deviation; "
                    
                    # Rule 2: Large quantity
                    df.loc[df['Quantity_Ordered'] > qty_threshold, 'Final_Decision'] = "PENDING_REVIEW"
                    df.loc[df['Quantity_Ordered'] > qty_threshold, 'Approval_Notes'] += "Large quantity order; "
                    
                    # Rule 3: New vendors (simplified)
                    if new_vendor_flag:
                        vendor_order_counts = df['Vendor_Name'].value_counts()
                        new_vendors = vendor_order_counts[vendor_order_counts < 3].index
                        df.loc[df['Vendor_Name'].isin(new_vendors), 'Final_Decision'] = "PENDING_REVIEW"
                        df.loc[df['Vendor_Name'].isin(new_vendors), 'Approval_Notes'] += "New vendor; "
                    
                    # Rule 4: Vendor rating (simulated)
                    vendor_ratings = df.groupby('Vendor_Name')['PO_Amount'].mean().rank(pct=True) * 5
                    low_rated_vendors = vendor_ratings[vendor_ratings < vendor_rating_threshold].index
                    df.loc[df['Vendor_Name'].isin(low_rated_vendors), 'Final_Decision'] = "PENDING_REVIEW"
                    df.loc[df['Vendor_Name'].isin(low_rated_vendors), 'Approval_Notes'] += "Low-rated vendor; "
                    
                    # Add AI predictions if available
                    if predictions is not None:
                        df['AI_Recommendation'] = predictions
                        df.loc[predictions == "Reject", 'Final_Decision'] = "PENDING_REVIEW"
                        df.loc[predictions == "Reject", 'Approval_Notes'] += "AI recommendation to reject; "
                    
                    # IMPORTANT: Save to session state for alerts
                    st.session_state.df = df.copy()
                    
                    # Display results
                    approval_stats = df['Final_Decision'].value_counts(normalize=True) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Auto-Approved", f"{approval_stats.get('APPROVED', 0):.1f}%")
                    col2.metric("Needs Review", f"{approval_stats.get('PENDING_REVIEW', 0):.1f}%")
                    
                    st.subheader("Approval Recommendations")
                    st.dataframe(
                        df[['Material_Name', 'Vendor_Name', 'Quantity_Ordered', 
                            'Unit_Price_Latest', 'Final_Decision', 
                            'Approval_Notes']].sort_values('Final_Decision'),
                        use_container_width=True
                    )
                    
                    # Generate approval workflow visualization
                    st.subheader("Approval Workflow")
                    fig = px.sunburst(
                        df,
                        path=['Final_Decision', 'Vendor_Name'],
                        values='PO_Amount',
                        title="Approval Status by Vendor"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Approval analysis failed: {str(e)}")
    
    with tab2:
        st.subheader("Order Optimization Engine")
        
        if st.button("Generate Order Optimization Recommendations"):
            with st.spinner("Optimizing procurement orders..."):
                try:
                    # Group by material and vendor to find optimization opportunities
                    optimized_orders = df.groupby(['Material_Name', 'Vendor_Name']).agg({
                        'Quantity_Ordered': ['sum', 'count', 'mean'],
                        'Unit_Price_Latest': ['mean', 'min'],
                        'PO_Amount': 'sum'
                    }).reset_index()
                    
                    optimized_orders.columns = [
                        'Material', 'Vendor', 'Total_Quantity', 'Order_Count',
                        'Avg_Order_Size', 'Avg_Price', 'Min_Price', 'Total_Spend'
                    ]
                    
                    # Calculate potential savings
                    optimized_orders['Potential_Savings'] = (
                        (optimized_orders['Avg_Price'] - optimized_orders['Min_Price']) * 
                        optimized_orders['Total_Quantity']
                    )
                    
                    # Identify optimization opportunities
                    optimized_orders['Recommendation'] = np.where(
                        optimized_orders['Order_Count'] > 3,
                        "Consolidate orders",
                        "Negotiate better price"
                    )
                    
                    optimized_orders.loc[
                        (optimized_orders['Order_Count'] > 3) & 
                        (optimized_orders['Avg_Order_Size'] < optimized_orders['Total_Quantity']/optimized_orders['Order_Count']),
                        'Recommendation'
                    ] = "Increase order size"
                    
                    # Display results
                    st.subheader("Top Optimization Opportunities")
                    st.dataframe(
                        optimized_orders.sort_values('Potential_Savings', ascending=False).head(10),
                        use_container_width=True
                    )
                    
                    # Visualize savings potential
                    fig = px.bar(
                        optimized_orders.nlargest(5, 'Potential_Savings'),
                        x='Material',
                        y='Potential_Savings',
                        color='Vendor',
                        title="Top 5 Potential Savings Opportunities"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Order optimization failed: {str(e)}")
    
    with tab3:
        st.subheader("Exception Management")
        
        if st.button("Identify Exception Cases"):
            with st.spinner("Analyzing exception cases..."):
                try:
                    # Identify exceptions based on multiple criteria
                    exceptions = df[
                        (df.get('Price_Deviation (%)', pd.Series()).abs() > df.get('Price_Deviation (%)', pd.Series()).quantile(0.9)) |
                        (df['Quantity_Ordered'] > df['Quantity_Ordered'].quantile(0.9)) |
                        (df['PO_Amount'] > df['PO_Amount'].quantile(0.9))
                    ].copy()
                    
                    if exceptions.empty:
                        st.success("No significant exceptions found in current data")
                        return
                    
                    # Categorize exceptions
                    exceptions['Exception_Type'] = np.where(
                        exceptions.get('Price_Deviation (%)', pd.Series()).abs() > df.get('Price_Deviation (%)', pd.Series()).quantile(0.9),
                        "Price Exception",
                        np.where(
                            exceptions['Quantity_Ordered'] > df['Quantity_Ordered'].quantile(0.9),
                            "Quantity Exception",
                            "Value Exception"
                        )
                    )
                    
                    # Display results
                    st.subheader("Exception Cases Requiring Attention")
                    st.dataframe(
                        exceptions.sort_values('PO_Amount', ascending=False),
                        use_container_width=True
                    )
                    
                    # Exception analysis visualization
                    fig = px.scatter(
                        exceptions,
                        x='Quantity_Ordered',
                        y='Unit_Price_Latest',
                        size='PO_Amount',
                        color='Exception_Type',
                        hover_name='Material_Name',
                        title="Exception Case Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Exception analysis failed: {str(e)}")

    # Initialize session state for DataFrame persistence
    if 'df' not in st.session_state:
        st.session_state.df = df.copy()

    st.markdown("---")
    st.subheader("🚨 Alert Notification Center")

    # Check if approval analysis has been run - now with session state
    approval_ready = 'Final_Decision' in st.session_state.df.columns

    with st.expander("⚙️ Alert Configuration", expanded=True):
        alert_recipients = st.text_input(
            "Notification Recipients (comma-separated emails)",
            value="procurement-team@example.com, manager@example.com"
        )
        
        alert_threshold = st.slider(
            "Price Deviation Alert Threshold (%)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0
        )
        
        send_alerts = st.checkbox("Enable email notifications", value=True)

    if not approval_ready:
        st.warning("⚠️ Please run the 'Approval Analysis' in the Approval Workflow tab first!")
        st.info("After running the analysis, return here to generate alerts")
    else:
        if st.button("Generate and Send Alerts"):
            with st.spinner("Generating alerts..."):
                try:
                    # Look for PENDING_REVIEW items
                    critical_alerts = st.session_state.df[
                        st.session_state.df['Final_Decision'] == "PENDING_REVIEW"
                    ].copy()
                    
                    # If you have Price_Deviation column, add the threshold filter
                    if 'Price_Deviation (%)' in st.session_state.df.columns:
                        critical_alerts = critical_alerts[
                            critical_alerts['Price_Deviation (%)'].abs() > alert_threshold
                        ]
                    
                    # Calculate total amount using PO_Amount if available
                    if 'PO_Amount' in st.session_state.df.columns:
                        critical_alerts['Total_Amount'] = critical_alerts['PO_Amount']
                    else:
                        critical_alerts['Total_Amount'] = critical_alerts['Quantity_Ordered'] * critical_alerts['Unit_Price_Latest']
                    
                    if not critical_alerts.empty:
                        alert_message = [
                            "🚨 Procurement Alerts Notification",
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            f"Total alerts: {len(critical_alerts)}",
                            f"Total value requiring review: ${critical_alerts['Total_Amount'].sum():,.2f}",
                            "",
                            "🔴 Items Requiring Immediate Review:"
                        ]
                        
                        # Add top 10 most expensive alerts (or all if less than 10)
                        top_alerts = critical_alerts.nlargest(min(10, len(critical_alerts)), 'Total_Amount')
                        
                        for _, row in top_alerts.iterrows():
                            alert_message.append(
                                f"▸ {row['Material_Name']} (Vendor {row['Vendor_Name']})\n"
                                f"   - Quantity: {row['Quantity_Ordered']} units\n"
                                f"   - Unit Price: ${row['Unit_Price_Latest']:.2f}\n"
                                f"   - Total Amount: ${row['Total_Amount']:,.2f}\n"
                                f"   - Status: {row['Final_Decision']}\n"
                                f"   - Notes: {row.get('Approval_Notes', 'No notes')}\n"
                                f"{'-'*40}"
                            )
                        
                        alert_content = "\n".join(alert_message)
                        
                        # Show preview
                        st.subheader("📋 Alert Preview")
                        st.text_area("Message Content", alert_content, height=400)
                        
                        # Show summary stats
                        st.subheader("📊 Alert Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Items Pending Review", len(critical_alerts))
                        with col2:
                            st.metric("Total Value", f"${critical_alerts['Total_Amount'].sum():,.2f}")
                        with col3:
                            st.metric("Average Order Value", f"${critical_alerts['Total_Amount'].mean():,.2f}")
                        
                        if send_alerts:
                            recipients = [email.strip() for email in alert_recipients.split(",")]
                            email_sent = send_email_alert(
                                recipients=recipients,
                                subject="🚨 Procurement Alerts Notification",
                                message=alert_content
                            )
                            if email_sent:
                                st.success(f"✉️ Alerts sent successfully to {', '.join(recipients)}")
                            else:
                                st.error("Failed to send email alerts")
                        else:
                            st.error("Failed to send alerts via email. Please check your SMTP configuration.")
                    else:
                        st.info("✅ No items currently pending review or meeting threshold criteria")
                        
                except KeyError as e:
                    st.error(f"Missing required data column: {str(e)}")
                    st.write("Available columns:", st.session_state.df.columns.tolist())
                except Exception as e:
                    st.error(f"Alert generation failed: {str(e)}")
                    # Debug info
                    st.write("Debug - DataFrame info:")
                    st.write("Shape:", st.session_state.df.shape)
                    st.write("Columns:", st.session_state.df.columns.tolist())
                    if not st.session_state.df.empty:
                        st.write("Sample data:", st.session_state.df.head(2))

    # Generate decision layer output for downstream systems
    if st.button("Generate Final Decisions"):
        with st.spinner("Compiling final procurement decisions..."):
            try:
                # Create consolidated decision output
                decisions = df[['Material_Name', 'Vendor_Name', 'Quantity_Ordered', 
                              'Unit_Price_Latest', 'PO_Amount']].copy()
                
                # Add AI recommendations if available
                if predictions is not None:
                    decisions['AI_Recommendation'] = predictions
                
                # FIXED: Use consistent decision values
                decisions['Final_Decision'] = np.where(
                    (decisions['Unit_Price_Latest'] <= decisions['Unit_Price_Latest'].median()) &
                    (decisions['Vendor_Name'].isin(df['Vendor_Name'].value_counts().nlargest(10).index)),
                    "APPROVED",
                    "PENDING_REVIEW"
                )
                
                # Format for export
                decisions['Decision_Date'] = datetime.now().strftime("%Y-%m-%d")
                decisions['Decision_ID'] = [f"DEC-{i:05d}" for i in range(1, len(decisions)+1)]
                
                # Update session state
                st.session_state.df = decisions.copy()
                
                st.success("Decision output generated successfully!")
                
                # Display sample
                st.subheader("Sample Decision Output")
                st.dataframe(decisions.head(10), use_container_width=True)
                
                # Export options
                st.download_button(
                    label="Download Decisions as CSV",
                    data=decisions.to_csv(index=False).encode('utf-8'),
                    file_name=f"procurement_decisions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
            
            except Exception as e:
                st.error(f"Decision compilation failed: {str(e)}")

def main():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
    
    # Load data and models
    df = load_data()
    scaler, pca, llm_model, xgb_model = load_models()
    pinecone_index = init_pinecone()
    
    # Check if data loaded successfully
    if df is None or df.empty:
        st.error("Failed to load data. Please check your data source.")
        return
    
    processed_df = preprocess_data(df)

    # Sidebar setup
    try:
        st.sidebar.image(Image.open(r"D:\Procurement\element_black-font-logo.png"), 
                       width=200, 
                       caption="Procurement Analytics")
    except FileNotFoundError:
        st.sidebar.warning("Icon image not found")
    
    # Navigation
    app_mode = st.sidebar.radio(
        "Navigation", 
        [
            "Dashboard Overview", 
            "AI Similar Material Pricing",
            "Predictive Analytics",
            "Vendor Intelligence",
            "Savings Optimizer",
            "AI Recommendation & Negotiation Agent",
            "Procurement Assistant",
            "Decision Layer"
        ],
        key="navigation"
    )
    
    # Main content area
    with st.container():
        # Route to appropriate section - ensure none of these functions return Streamlit elements
        if app_mode == "Dashboard Overview":
            procurement_dashboard(df)
        elif app_mode == "AI Similar Material Pricing":
            ai_powered_search_section(df, processed_df, scaler, pca, llm_model, pinecone_index)
        elif app_mode == "Predictive Analytics":
            predictive_analytics(df, xgb_model)
        elif app_mode == "Vendor Intelligence":
            vendor_intelligence(df)
        elif app_mode == "Savings Optimizer":
            spend_optimizer(df)
        elif app_mode == "AI Recommendation & Negotiation Agent":
            procurement_recommendation_agent(df)
        elif app_mode == "Procurement Assistant":
            procurement_assistant(df)
        elif app_mode == "Decision Layer":
        # Pass predictions if available from previous layers
            predictions = None
            if 'predictions' in st.session_state:
                predictions = st.session_state.predictions
            decision_layer(df, predictions)


if __name__ == "__main__":
    main()