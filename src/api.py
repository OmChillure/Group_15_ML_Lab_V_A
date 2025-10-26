from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from wallet_validator import WalletMLValidator

app = FastAPI(
    title="Wallet Validator API",
    version="1.0",
    description="ML-powered blockchain wallet address validation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

validator = WalletMLValidator()
model_loaded = False

@app.on_event("startup")
async def startup_event():
    global model_loaded
    model_paths = [
        'src/wallet_validator.pkl',
        'wallet_validator.pkl',
        'src/wallet_validator_model.pkl',
        'wallet_validator_model.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                validator.load_model(path)
                model_loaded = True
                print(f"Model loaded from {path}")
                break
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    if not model_loaded:
        print(" Warning: Model not loaded. Train first!")


class WalletRequest(BaseModel):
    network: str
    address: str
    
    class Config:
        schema_extra = {
            "example": {
                "network": "evm",
                "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1"
            }
        }


class WalletResponse(BaseModel):
    is_valid: bool
    claimed_network: str
    predicted_network: str
    confidence: float
    
    class Config:
        schema_extra = {
            "example": {
                "is_valid": True,
                "claimed_network": "evm",
                "predicted_network": "evm",
                "confidence": 0.98
            }
        }


@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "Wallet Validator API",
        "status": "active",
        "version": "1.0"
    }


@app.post("/validate", response_model=WalletResponse)
def validate_wallet(request: WalletRequest):
    """
    Validate if wallet address matches claimed network
    
    **Parameters:**
    - network: Network type (evm, bitcoin, solana)
    - address: Wallet address string
    
    **Returns:**
    - is_valid: Whether address matches claimed network
    - predicted_network: ML model's prediction
    - confidence: Prediction confidence (0-1)
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        result = validator.predict(request.network, request.address)
        return WalletResponse(
            is_valid=result['is_valid'],
            claimed_network=result['claimed_network'],
            predicted_network=result['predicted_network'],
            confidence=result['confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/networks")
def get_networks():
    """Get list of supported networks"""
    return {
        "networks": ["evm", "bitcoin", "solana"],
        "descriptions": {
            "evm": "Ethereum and EVM-compatible chains",
            "bitcoin": "Bitcoin (Legacy, SegWit, Bech32)",
            "solana": "Solana blockchain"
        }
    }


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
