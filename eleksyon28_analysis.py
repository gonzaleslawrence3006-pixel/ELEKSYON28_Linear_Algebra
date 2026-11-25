import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42
FEATURES = [
    "previous_election_turnout_pct",
    "registered_voters",
    "voting_age_population",
    "population_growth_rate_pct"
]
TARGET = "actual_voter_turnout_pct"

# Part A: Load Dataset & Train/Test Split
df = pd.read_csv("ph_voter_turnout_dataset.csv")
X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)

# 75% Train (330 rows) / 25% Test (110 rows)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)
n, p = X_train.shape

print("--- PART A: DATASET LOADING & TRAIN/TEST SPLIT ---")
print("Total dataset shape:", df.shape)
print("Input matrix X shape:", X.shape)
print("Target vector y shape:", y.shape)
print("X_train shape:", X_train.shape, "| y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "| y_test shape:", y_test.shape)
print("\nFirst 5 rows of features (X) and target (y):")
print(df[FEATURES + [TARGET]].head())

# Part B & C: Matrix dimensions and Rank
print("\n--- PARTS B & C: MATRICES & RANK ---")
print("X_train transpose shape:", X_train.T.shape)
print("Raw Gram matrix (X_train.T @ X_train):\n", X_train.T @ X_train)
print("Rank of X_train:", np.linalg.matrix_rank(X_train))

# Part D: Standardization (fit on training data only)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
M = np.diag(1.0 / scaler.scale_)

print("\n--- PART D: STANDARDIZATION ---")
print("Training means (mu):", scaler.mean_)
print("Training stds (sigma):", scaler.scale_)
print("det(M):", np.linalg.det(M))

# Part E: Standardized Gram Matrix, Determinant & Conditioning
G = X_train_std.T @ X_train_std
print("\n--- PART E: GRAM MATRIX & CONDITIONING ---")
print("det(G):", np.linalg.det(G))
print("Condition number of G:", np.linalg.cond(G))
print("Rank of G:", np.linalg.matrix_rank(G))

# Part F: Covariance Eigendecomposition
C = np.cov(X_train_std, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(C)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(variance_ratio)

print("\n--- PART F: EIGENVALUES & EIGENVECTORS ---")
print("Eigenvalues:", eigenvalues)
print("Explained variance ratio (%):", np.round(variance_ratio * 100, 2))
print("Cumulative variance (%):", np.round(cumulative_variance * 100, 2))

# Part G: Diagonalization & PCA Projection
V = eigenvectors
Lambda = np.diag(eigenvalues)
print("\n--- PART G: DIAGONALIZATION & PCA ---")
print("Reconstruction error ||C - V*Lambda*V.T||:", np.linalg.norm(C - (V @ Lambda @ V.T), ord="fro"))

k = int(np.argmax(cumulative_variance >= 0.95) + 1)
V_k = V[:, :k]
X_train_pca = X_train_std @ V_k
X_test_pca = X_test_std @ V_k
print(f"Components needed for >=95% variance: k = {k} ({cumulative_variance[k-1]*100:.2f}%)")

# Part H: Ordinary Least Squares (OLS)
def fit_ols(A, target):
    A_bias = np.column_stack([np.ones(A.shape[0]), A])
    beta, _, _, _ = np.linalg.lstsq(A_bias, target, rcond=None)
    return beta

def predict(A, beta):
    A_bias = np.column_stack([np.ones(A.shape[0]), A])
    return A_bias @ beta

beta_full = fit_ols(X_train_std, y_train)
pred_full = predict(X_test_std, beta_full)

beta_pca = fit_ols(X_train_pca, y_train)
pred_pca = predict(X_test_pca, beta_pca)

print("\n--- PART H: MODEL EVALUATION ---")
print(f"Full OLS (4 features) -> RMSE: {np.sqrt(mean_squared_error(y_test, pred_full)):.4f}, R2: {r2_score(y_test, pred_full):.4f}")
print(f"PCA OLS (k={k})         -> RMSE: {np.sqrt(mean_squared_error(y_test, pred_pca)):.4f}, R2: {r2_score(y_test, pred_pca):.4f}")
print("First test sample -> Actual:", y_test[0], "| Predicted (Full OLS):", round(pred_full[0], 4))

# Save the 3 required figures
plt.figure(figsize=(6, 4))
plt.plot(range(1, p + 1), variance_ratio, marker="o")
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("Scree Plot")
plt.xticks(range(1, p + 1))
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("scree_plot.png", dpi=180)
plt.show()

plt.figure(figsize=(6, 4))
pts = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="viridis", alpha=0.85)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Training Data Projected onto PC1 and PC2")
plt.colorbar(pts, label="Voter Turnout (%)")
plt.tight_layout()
plt.savefig("pca_projection.png", dpi=180)
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred_full, alpha=0.85)
low, high = min(y_test.min(), pred_full.min()), max(y_test.max(), pred_full.max())
plt.plot([low, high], [low, high], "r--", label="Ideal prediction")
plt.xlabel("Actual Voter Turnout (%)")
plt.ylabel("Predicted Voter Turnout (%)")
plt.title("Predicted vs Actual (Full OLS Model)")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=180)
plt.show()

# ==============================================================================
# EXPECTED SYSTEM FEATURES (1, 2, & 3)
# ==============================================================================

# 1. Historical Election Data Management (Store & summarize historical stats)
hist_summary = df.groupby("election_year").agg(
    Municipalities=("municipality", "count"),
    Avg_Turnout_Pct=("actual_voter_turnout_pct", "mean"),
    Total_Registered_Voters=("registered_voters", "sum")
).round(2).reset_index()

print("\n--- SYSTEM FEATURE 1: HISTORICAL ELECTION DATA SUMMARY ---")
print(hist_summary.to_string(index=False))
hist_summary.to_csv("historical_election_summary.csv", index=False)

# 2. Voter Turnout Prediction (Forecast 2028 using 2025 latest baseline)
df_2025 = df[df["election_year"] == 2025].copy()
growth_3yr = (1 + (df_2025["population_growth_rate_pct"] / 100.0)) ** 3

X_2028 = np.column_stack([
    df_2025["actual_voter_turnout_pct"].values,      # 2025 actual becomes 2028 previous turnout
    df_2025["registered_voters"].values * growth_3yr,
    df_2025["voting_age_population"].values * growth_3yr,
    df_2025["population_growth_rate_pct"].values
])
X_2028_std = scaler.transform(X_2028)
df_2025["Predicted_2028_Turnout_Pct"] = np.round(np.clip(predict(X_2028_std, beta_full), 0, 100), 2)

# 3. Geographical Prediction (By Region and City/Municipality)
geo_region = df_2025.groupby("region").agg(
    Cities_Tracked=("municipality", "count"),
    Turnout_2025_Pct=("actual_voter_turnout_pct", "mean"),
    Predicted_2028_Pct=("Predicted_2028_Turnout_Pct", "mean")
).round(2).reset_index()

print("\n--- SYSTEM FEATURES 2 & 3: 2028 GEOGRAPHICAL PREDICTION BY REGION ---")
print(geo_region.to_string(index=False))

geo_city = df_2025[["region", "municipality", "actual_voter_turnout_pct", "Predicted_2028_Turnout_Pct"]].rename(
    columns={"actual_voter_turnout_pct": "Turnout_2025_Pct"}
)
geo_city.to_csv("predictions_2028_by_city.csv", index=False)
geo_region.to_csv("predictions_2028_by_region.csv", index=False)
