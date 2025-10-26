from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from wallet_validator import WalletMLValidator
import uvicorn

app = FastAPI(title="Wallet Validator API", version="1.0")

# Load model on startup
validator = WalletMLValidator()
try:
    validator.load_model('wallet_validator_model.pkl')
except:
    print("Warning: Model not loaded. Train first!")

class WalletRequest(BaseModel):
    network: str
    address: str

class WalletResponse(BaseModel):
    is_valid: bool
    claimed_network: str
    predicted_network: str
    confidence: float

@app.get("/")
def root():
    return {"message": "Wallet Validator API", "status": "active"}

@app.post("/validate", response_model=WalletResponse)
def validate_wallet(request: WalletRequest):
    """Validate if wallet address matches claimed network"""
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

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": validator.model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
