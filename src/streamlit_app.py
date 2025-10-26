import streamlit as st
import pandas as pd
from wallet_validator import WalletMLValidator
import plotly.graph_objects as go

st.set_page_config(page_title="Wallet Validator", page_icon="🔐", layout="wide")

# Initialize session state
if 'validator' not in st.session_state:
    st.session_state.validator = WalletMLValidator()
    try:
        st.session_state.validator.load_model('wallet_validator_model.pkl')
        st.session_state.model_loaded = True
    except:
        st.session_state.model_loaded = False

st.title("🔐 Blockchain Wallet Validator")
st.markdown("ML-powered wallet address validation across multiple networks")

# Sidebar for training
with st.sidebar:
    st.header("⚙️ Model Training")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ No model found")
    
    st.markdown("---")
    st.subheader("Train New Model")
    
    uploaded_files = st.file_uploader(
        "Upload CSV files (network,address)",
        accept_multiple_files=True,
        type=['csv']
    )
    
    if uploaded_files and st.button("Train Model"):
        with st.spinner("Training model..."):
            dfs = [pd.read_csv(f) for f in uploaded_files]
            combined = pd.concat(dfs, ignore_index=True)
            
            st.session_state.validator.train(combined)
            st.session_state.validator.save_model('wallet_validator_model.pkl')
            st.session_state.model_loaded = True
            st.success("Model trained successfully!")
            st.rerun()

# Main content
if not st.session_state.model_loaded:
    st.error("⚠️ Please train a model first using the sidebar")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["🔍 Single Validation", "📊 Batch Validation"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        network = st.selectbox(
            "Select Network",
            ["evm", "bitcoin", "solana"],
            key="network_select"
        )
    
    with col2:
        address = st.text_input(
            "Enter Wallet Address",
            placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1",
            key="address_input"
        )
    
    if st.button("Validate", type="primary", use_container_width=True):
        if address:
            result = st.session_state.validator.predict(network, address)
            
            # Display result
            if result['is_valid']:
                st.success(f"✅ Valid {network.upper()} address")
            else:
                st.error(f"❌ Invalid! This appears to be a {result['predicted_network'].upper()} address")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Claimed Network", result['claimed_network'].upper())
            col2.metric("Predicted Network", result['predicted_network'].upper())
            col3.metric("Confidence", f"{result['confidence']:.1%}")
            
            # Probability chart
            st.subheader("Network Probabilities")
            probs = result['all_probabilities']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=['green' if k == result['predicted_network'] else 'lightgray' for k in probs.keys()]
                )
            ])
            fig.update_layout(
                yaxis_title="Probability",
                xaxis_title="Network",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter a wallet address")

with tab2:
    st.subheader("Batch Validation")
    
    batch_file = st.file_uploader(
        "Upload CSV for batch validation (network,address)",
        type=['csv'],
        key="batch_upload"
    )
    
    if batch_file:
        df = pd.read_csv(batch_file)
        
        if st.button("Validate Batch", type="primary"):
            with st.spinner("Validating..."):
                results = []
                for _, row in df.iterrows():
                    result = st.session_state.validator.predict(row['network'], row['address'])
                    results.append({
                        'Network': row['network'],
                        'Address': row['address'][:20] + '...',
                        'Valid': '✅' if result['is_valid'] else '❌',
                        'Predicted': result['predicted_network'],
                        'Confidence': f"{result['confidence']:.1%}"
                    })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary
                valid_count = sum(1 for r in results if r['Valid'] == '✅')
                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(results))
                col2.metric("Valid", valid_count)
                col3.metric("Invalid", len(results) - valid_count)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "validation_results.csv",
                    "text/csv",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("*Powered by Machine Learning - No regex used*")
