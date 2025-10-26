# Blockchain Wallet Address Validator

**ML-powered wallet address validation across multiple blockchain networks**

---

## Overview

This project uses **machine learning** instead of traditional regex patterns to validate cryptocurrency wallet addresses. The model learns character-level patterns and statistical features to accurately classify addresses across:

- **EVM (Ethereum)** - Ethereum and EVM-compatible chains
- **Bitcoin** - Legacy, SegWit, and Bech32 formats  
- **Solana** - Solana blockchain addresses

### Why ML over Regex?

**Handles edge cases** automatically  
**Adapts to new address formats** with retraining  
**Provides confidence scores** for predictions  
**More maintainable** than complex regex patterns  

---

## Features

- **Gradient Boosting Classifier** 
- **Interactive Streamlit UI** for single/batch validation
- **FastAPI REST API** for production integration
- **Comprehensive EDA** with visualizations
- **Model comparison** framework (ML vs Regex)

---

## Project Structure

```
wallet-validator/
├── datasets/
│   ├── ethwallets.csv          # EVM addresses
│   ├── bitcoinwallets.csv      # Bitcoin addresses
│   └── solanawallets.csv       # Solana addresses
├── artifacts/
│   └── eda/                    # EDA visualizations
├── wallet_validator.py         # Core ML model
├── eda_.py             # Exploratory data analysis
├── train_model.py              # Model training script
├── streamlit_app.py            # Web UI
├── api.py                      # FastAPI backend
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Installation

### Prerequisites
- Python 3.10
- pip package manager

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd <folder>
```

### Step 2: Virtual environment & Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Prepare Data
Place your CSV files in `datasets/` folder with columns:
- `network` - Network type (evm, bitcoin, solana)
- `address` - Wallet address string

---

## Usage

### 1. Run EDA (Exploratory Data Analysis)
```bash
cd src
python eda_analysis.py

or 

python src/eda_analysis.py
```
**Output:** Generates visualizations in `artifacts/eda/`

### 2. Train Model
```bash
cd src
python train_model.py

or 

python src/train_model.py
```
**Output:** 
- Trained model saved as `wallet_validator_model.pkl`
- Training metrics printed to console

### 3. Launch Streamlit UI
```bash
streamlit run streamlit_app.py
```
**Access:** Open browser at `http://localhost:8501`

### 4. Start FastAPI Server
```bash
cd src
python api.py

or 

python src/api.py
```
**Access:** API at `http://localhost:8000`  
**Docs:** Swagger UI at `http://localhost:8000/docs`

### 5. Programmatic Usage
```python
from wallet_validator import WalletMLValidator

# Load trained model
validator = WalletMLValidator()
validator.load_model('wallet_validator_model.pkl')

# Validate address
result = validator.predict('evm', '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1')

print(f"Valid: {result['is_valid']}")
print(f"Predicted: {result['predicted_network']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## Model Performance

### Training Results
| Network  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| EVM      | 0.99      | 1.00   | 0.99     | 2000    |
| Bitcoin  | 0.98      | 0.97   | 0.98     | 2000    |
| Solana   | 0.99      | 0.99   | 0.99     | 2000    |
| **Avg**  | **0.99**  | **0.99** | **0.99** | **6000** |

### Model Comparison
| Model              | Accuracy | Training Time | Inference Speed |
|--------------------|----------|---------------|-----------------|
| Regex Baseline     | 89.3%    | N/A           | 0.001ms        |
| Random Forest      | 97.2%    | 2.3s          | 0.8ms          |
| **Gradient Boost** | **98.7%**| **3.1s**      | **1.2ms**      |

---

## Key Findings

### From EDA
1. **EVM addresses**: Always 42 characters, start with `0x`, hexadecimal
2. **Bitcoin addresses**: 26-62 chars, start with `1`, `3`, or `bc1`
3. **Solana addresses**: ~44 characters, base58 encoding (no 0, O, I, l)


### Model Insights
- Gradient Boosting outperforms Random Forest by 1.5%
- Cross-validation
- No overfitting detected (train/test gap <1%)
- Handles malformed addresses better than regex

---

## Requirements

```
pandas
numpy
scikit-learn
streamlit
fastapi
uvicorn
plotly
matplotlib
seaborn
```

---

## Future Enhancements

- [ ] Add support for more networks (Cardano, Polkadot, etc.)
- [ ] Implement checksum validation
- [ ] Deploy as containerized service (Docker)
- [ ] Add monitoring and logging dashboard
- [ ] Integrate with blockchain APIs for live validation

---
