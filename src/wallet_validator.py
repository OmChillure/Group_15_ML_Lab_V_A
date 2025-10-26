import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class WalletMLValidator:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

    def extract_features(self, address):
        """Extract character-level and statistical features from address"""
        features = {}

        features["length"] = len(address)

        features["digit_count"] = sum(c.isdigit() for c in address)
        features["lower_count"] = sum(c.islower() for c in address)
        features["upper_count"] = sum(c.isupper() for c in address)
        features["special_count"] = sum(not c.isalnum() for c in address)

        features["digit_ratio"] = features["digit_count"] / len(address)
        features["lower_ratio"] = features["lower_count"] / len(address)
        features["upper_ratio"] = features["upper_count"] / len(address)
        features["special_ratio"] = features["special_count"] / len(address)

        for i in range(min(5, len(address))):
            char = address[i]
            features[f"pos{i}_digit"] = 1 if char.isdigit() else 0
            features[f"pos{i}_lower"] = 1 if char.islower() else 0
            features[f"pos{i}_upper"] = 1 if char.isupper() else 0
            features[f"pos{i}_special"] = 1 if not char.isalnum() else 0
            features[f"pos{i}_ascii"] = ord(char) / 127.0

        for i in range(len(address), 5):
            features[f"pos{i}_digit"] = 0
            features[f"pos{i}_lower"] = 0
            features[f"pos{i}_upper"] = 0
            features[f"pos{i}_special"] = 0
            features[f"pos{i}_ascii"] = 0

        for i in range(min(3, len(address))):
            char = address[-(i + 1)]
            features[f"last{i}_digit"] = 1 if char.isdigit() else 0
            features[f"last{i}_lower"] = 1 if char.islower() else 0
            features[f"last{i}_upper"] = 1 if char.isupper() else 0

        for i in range(len(address), 3):
            features[f"last{i}_digit"] = 0
            features[f"last{i}_lower"] = 0
            features[f"last{i}_upper"] = 0

        features["unique_chars"] = len(set(address))
        features["char_diversity"] = len(set(address)) / len(address)


        max_consecutive_digits = 0
        max_consecutive_letters = 0
        current_digits = 0
        current_letters = 0

        for char in address:
            if char.isdigit():
                current_digits += 1
                current_letters = 0
                max_consecutive_digits = max(max_consecutive_digits, current_digits)
            elif char.isalpha():
                current_letters += 1
                current_digits = 0
                max_consecutive_letters = max(max_consecutive_letters, current_letters)
            else:
                current_digits = 0
                current_letters = 0

        features["max_consecutive_digits"] = max_consecutive_digits
        features["max_consecutive_letters"] = max_consecutive_letters

        return features

    def prepare_data(self, df):
        """Convert dataframe to feature matrix"""
        feature_list = []
        for _, row in df.iterrows():
            features = self.extract_features(row["address"])
            feature_list.append(features)

        X = pd.DataFrame(feature_list)
        y = self.label_encoder.fit_transform(df["network"])

        return X, y

    def train(self, df, test_size=0.2):
        """Train the model"""
        X, y = self.prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42
        )

        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        y_pred = self.model.predict(X_test)

        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_
            )
        )
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return self

    def predict(self, network, address):
        """Predict if the address matches the claimed network"""
        features = self.extract_features(address)
        X = pd.DataFrame([features])

        predicted_network_idx = self.model.predict(X)[0]
        predicted_network = self.label_encoder.inverse_transform(
            [predicted_network_idx]
        )[0]

        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities[predicted_network_idx]

        is_valid = predicted_network.lower() == network.lower()

        return {
            "is_valid": is_valid,
            "claimed_network": network,
            "predicted_network": predicted_network,
            "confidence": confidence,
            "all_probabilities": dict(zip(self.label_encoder.classes_, probabilities)),
        }

    def save_model(self, filepath="wallet_validator_model.pkl"):
        """Save trained model"""
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)

    def load_model(self, filepath="wallet_validator_model.pkl"):
        """Load trained model"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.label_encoder = data["label_encoder"]


# Example usage
if __name__ == "__main__":
    # Sample data generation (replace with your actual data)
    sample_data = {
        "network": ["evm"] * 5 + ["bitcoin"] * 5 + ["solana"] * 5,
        "address": [
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1",
            "0x1234567890abcdef1234567890abcdef12345678",
            "0xabcdefABCDEF1234567890abcdefABCDEF123456",
            "0x9876543210fedcba9876543210fedcba98765432",
            "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy",
            "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
            "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
            "3FZbgi29cpjq2GjdwV8eyHuJJnkLtktZc5",
            "7UX2i7SucgLMQcfZ75s3VXmZLa4r2HQm1Z1r8RJcuJN",
            "DYw8jCTfwHNRJhhmFcbXvVDTqWMEVjL6pZfnq9vE1pza",
            "9WzDXwBbmkg8ZTbNMqUxvQRAn26nQmXp9M3PY5Qvyq6t",
            "H6ARHf6YXhGYeQfUxcqR7wNNEeqRYVW5LkgjY8kx3qcJ",
            "5KJvsngHeMpm884wtkJNzQGaCErckhHJBGFsvd3VyK5n",
        ],
    }

    df = pd.DataFrame(sample_data)

    # Train model
    validator = WalletMLValidator()
    validator.train(df)

    # Save model
    validator.save_model()

    print("\n" + "=" * 60)
    print("Testing predictions:")
    print("=" * 60)

    # Test predictions
    test_cases = [
        (
            "ethereum",
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1",
        ),  # Change evm → ethereum
        ("bitcoin", "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1"),
        ("solana", "7UX2i7SucgLMQcfZ75s3VXmZLa4r2HQm1Z1r8RJcuJN"),
        ("ethereum", "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"),
    ]

    for network, address in test_cases:
        result = validator.predict(network, address)
        print(f"\nClaimed: {result['claimed_network']} | Address: {address[:30]}...")
        print(
            f"Valid: {result['is_valid']} | Predicted: {result['predicted_network']} | Confidence: {result['confidence']:.2%}"
        )
