import pandas as pd
from wallet_validator import WalletMLValidator
evm_df = pd.read_csv('../datasets/ethwallets.csv') 
btc_df = pd.read_csv('../datasets/bitcoinwallets.csv')
sol_df = pd.read_csv('../datasets/solanawallets.csv')

combined_df = pd.concat([evm_df, btc_df, sol_df], ignore_index=True)

print(f"Total wallets loaded: {len(combined_df)}")
print(f"Network distribution:\n{combined_df['network'].value_counts()}")

validator = WalletMLValidator()
validator.train(combined_df, test_size=0.2)

validator.save_model('wallet_validator.pkl')

print("\n" + "="*60)
print("Model trained and saved successfully!")
print("="*60)

print("\nTesting on sample inputs:")

# Test case 1: Valid EVM address
result = validator.predict('evm', '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1')
print(f"\nTest 1 - EVM address as EVM:")
print(f"  Valid: {result['is_valid']}")
print(f"  Predicted: {result['predicted_network']} (confidence: {result['confidence']:.2%})")

print("\n" + "="*60)
print("To use later, load the model:")
print("validator = WalletMLValidator()")
print("validator.load_model('wallet_validator_10k.pkl')")
print("="*60)