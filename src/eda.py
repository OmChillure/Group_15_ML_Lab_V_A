import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_style('whitegrid')
os.makedirs('artifacts/eda', exist_ok=True)

print(" WALLET ADDRESS VALIDATOR - EDA")
print("="*60)

print("\n1️⃣ Loading Data...")
evm_df = pd.read_csv('../datasets/ethwallets.csv')
btc_df = pd.read_csv('../datasets/bitcoinwallets.csv')
sol_df = pd.read_csv('../datasets/solanawallets.csv')

combined_df = pd.concat([evm_df, btc_df, sol_df], ignore_index=True)

print(f" Total addresses: {len(combined_df):,}")
print(f"   - EVM: {len(evm_df):,}")
print(f"   - Bitcoin: {len(btc_df):,}")
print(f"   - Solana: {len(sol_df):,}")


print("\n Data Quality...")
print(f"Missing values: {combined_df.isnull().sum().sum()}")
print(f"Duplicates: {combined_df.duplicated().sum()}")
print(f"Unique addresses: {combined_df['address'].nunique():,}")


print("\n Address Characteristics...")

# Add features
combined_df['length'] = combined_df['address'].apply(len)
combined_df['starts_0x'] = combined_df['address'].str.lower().str.startswith('0x').astype(int)
combined_df['digit_count'] = combined_df['address'].apply(lambda x: sum(c.isdigit() for c in x))
combined_df['lower_count'] = combined_df['address'].apply(lambda x: sum(c.islower() for c in x))
combined_df['upper_count'] = combined_df['address'].apply(lambda x: sum(c.isupper() for c in x))

print("\nLength by Network:")
print(combined_df.groupby('network')['length'].agg(['min', 'max', 'mean']))

print("\nPrefix '0x' by Network:")
print(combined_df.groupby('network')['starts_0x'].sum())


print("\n Creating Visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

network_counts = combined_df['network'].value_counts()
axes[0].bar(network_counts.index, network_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(network_counts.values):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')

axes[1].pie(network_counts.values, labels=network_counts.index, autopct='%1.1f%%', 
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], startangle=90)
axes[1].set_title('Network Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('artifacts/eda/fig1_target_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: fig1_target_distribution.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
for network in combined_df['network'].unique():
    data = combined_df[combined_df['network'] == network]['length']
    ax.hist(data, alpha=0.6, label=network.upper(), bins=30)
ax.set_xlabel('Address Length', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Address Length Distribution by Network', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('artifacts/eda/fig2_length_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: fig2_length_distribution.png")
plt.close()

sample_df = combined_df.groupby('network').sample(n=min(1000, len(combined_df)//3), random_state=RANDOM_SEED)
feature_cols = ['length', 'digit_count', 'lower_count', 'upper_count', 'starts_0x']
corr_matrix = sample_df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('artifacts/eda/fig3_feature_correlation.png', dpi=300, bbox_inches='tight')
print("Saved: fig3_feature_correlation.png")
plt.close()

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

examples = []
for network in ['evm', 'bitcoin', 'solana']:
    sample = combined_df[combined_df['network'] == network].head(3)
    for _, row in sample.iterrows():
        examples.append([network.upper(), row['address'][:50] + '...', row['length']])

table = ax.table(cellText=examples, colLabels=['Network', 'Address (truncated)', 'Length'],
                cellLoc='left', loc='center', colWidths=[0.15, 0.65, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(3):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Sample Wallet Addresses', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('artifacts/eda/fig4_example_records.png', dpi=300, bbox_inches='tight')
print("Saved: fig4_example_records.png")
plt.close()

print("\n EDA Completed. Visualizations saved in 'artifacts/eda/' directory.")