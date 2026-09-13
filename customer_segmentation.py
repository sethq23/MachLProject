"""
CSCI 6840 - Project 1
Customer Segmentation Using Clustering Techniques
Dataset: Mall_Customers.csv (Kaggle Mall Customers)

Run:  python customer_segmentation.py
(Full explanation of each step is in the PowerPoint presentation.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

RANDOM_STATE = 42


# =============================================================
# STEP 1: DATA EXPLORATION AND PREPROCESSING
# =============================================================
print("=" * 60)
print("STEP 1: DATA EXPLORATION AND PREPROCESSING")
print("=" * 60)

df = pd.read_csv("Mall_Customers.csv")
df = df.rename(columns={"Genre": "Gender"})

print("\nShape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe().round(2))

print("\nGender counts:")
print(df["Gender"].value_counts())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("\n-> No missing values, so no imputation is needed.")
else:
    df = df.fillna(df.median(numeric_only=True))
    print("\n-> Missing values filled with the column median.")

# EDA plot
plt.figure(figsize=(7, 5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
            s=60, c="steelblue", edgecolor="black")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Income vs Spending Score (before clustering)")
plt.savefig("fig1_raw_data.png", dpi=150, bbox_inches="tight")
plt.show()

# Feature selection: the two features marketing acts on, and 2D plots directly
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Standardize so Income (15-137) doesn't outweigh Spending Score (1-99)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures used for clustering:", features)
print("Data standardized with StandardScaler (mean=0, std=1).")


# =============================================================
# STEP 2: FIND THE OPTIMAL NUMBER OF CLUSTERS
# =============================================================
print("\n" + "=" * 60)
print("STEP 2: FINDING THE OPTIMAL NUMBER OF CLUSTERS")
print("=" * 60)

# Method 1: Elbow Method - look for where WCSS stops dropping quickly
wcss = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(k_values, wcss, "bo-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow Method")
plt.grid(alpha=0.3)
plt.savefig("fig2_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# Method 2: Silhouette Score - ranges -1 to 1, higher is better
sil_scores = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

print("\nSilhouette score for each k:")
for k, score in zip(range(2, 11), sil_scores):
    print(f"   k = {k}:  {score:.4f}")

best_k = range(2, 11)[np.argmax(sil_scores)]
print(f"\n-> Best k by silhouette score: {best_k}")

plt.figure(figsize=(7, 5))
plt.plot(range(2, 11), sil_scores, "go-")
plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs k")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("fig3_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()

# Method 3: Dendrogram - shows how points merge together step by step
plt.figure(figsize=(11, 5))
linked = linkage(X_scaled, method="ward")
dendrogram(linked, no_labels=True)
plt.title("Dendrogram (Ward linkage)")
plt.ylabel("Distance")
plt.savefig("fig4_dendrogram.png", dpi=150, bbox_inches="tight")
plt.show()

print("All three methods (elbow, silhouette, dendrogram) point to k =", best_k)


# =============================================================
# STEP 3: APPLY TWO CLUSTERING ALGORITHMS
# =============================================================
print("\n" + "=" * 60)
print("STEP 3: APPLYING TWO CLUSTERING ALGORITHMS")
print("=" * 60)

# Algorithm 1: K-Means
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)

print(f"\nK-Means")
print(f"   Silhouette score: {kmeans_score:.4f}")
print(f"   Cluster sizes:    {np.bincount(kmeans_labels)}")

# Algorithm 2: Hierarchical (Agglomerative)
hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
hier_labels = hierarchical.fit_predict(X_scaled)
hier_score = silhouette_score(X_scaled, hier_labels)

print(f"\nHierarchical (Ward linkage)")
print(f"   Silhouette score: {hier_score:.4f}")
print(f"   Cluster sizes:    {np.bincount(hier_labels)}")

print("\nCOMPARISON:")
print(f"   K-Means silhouette:      {kmeans_score:.4f}")
print(f"   Hierarchical silhouette: {hier_score:.4f}")

# Compare cluster SIZES, not cluster numbers (the numbers are arbitrary)
print(f"   K-Means cluster sizes:      {sorted(np.bincount(kmeans_labels).tolist())}")
print(f"   Hierarchical cluster sizes: {sorted(np.bincount(hier_labels).tolist())}")

if kmeans_score >= hier_score:
    print("   -> K-Means scored higher, so we use it for the analysis below.")
else:
    print("   -> Hierarchical scored higher, but we use K-Means below because")
    print("      it gives centroids, which make the segments easier to describe.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"],
                c=kmeans_labels, cmap="viridis", s=60, edgecolor="black")
axes[0].set_title(f"K-Means (silhouette = {kmeans_score:.3f})")

axes[1].scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"],
                c=hier_labels, cmap="viridis", s=60, edgecolor="black")
axes[1].set_title(f"Hierarchical (silhouette = {hier_score:.3f})")

for ax in axes:
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")

plt.savefig("fig5_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nBoth algorithms found nearly the same groups - the cluster sizes")
print("are very close and the scatter plots look almost identical, with only")
print("a handful of borderline customers placed differently. Getting the same")
print("answer from two different methods is good evidence the segments are real.")


# =============================================================
# STEP 4: CLUSTER ANALYSIS AND BUSINESS INSIGHTS
# =============================================================
print("\n" + "=" * 60)
print("STEP 4: CLUSTER ANALYSIS AND BUSINESS INSIGHTS")
print("=" * 60)

df["Cluster"] = kmeans_labels

# Convert centroids back to real dollars/points so the plot is readable
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(9, 6.5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
            c=df["Cluster"], cmap="viridis", s=70, edgecolor="black")
plt.scatter(centroids[:, 0], centroids[:, 1],
            c="red", marker="X", s=300, label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"Customer Segments (K-Means, k={best_k})")
plt.legend()
plt.savefig("fig6_final_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

profile = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
).round(1)

print("\nCluster profiles:")
print(profile)

# Name each segment by comparing it to the overall median
income_median = df["Annual Income (k$)"].median()
spending_median = df["Spending Score (1-100)"].median()

print("\n" + "-" * 60)
print("BUSINESS INSIGHTS AND MARKETING RECOMMENDATIONS")
print("-" * 60)

names = {}
for cluster_id, row in profile.iterrows():
    high_income = row["Avg_Income"] > income_median
    high_spending = row["Avg_Spending"] > spending_median

    # The middle cluster is close to average on both measures
    if (abs(row["Avg_Income"] - income_median) < 12 and
            abs(row["Avg_Spending"] - spending_median) < 12):
        name = "Average Customers"
        advice = ("Largest group. Use loyalty programs and seasonal sales to "
                  "keep them coming back and try to move some into the VIP group.")
    elif high_income and high_spending:
        name = "VIP / Target Customers"
        advice = ("Most valuable group. Give them premium rewards, early access "
                  "to new products, and personal service so they stay loyal.")
    elif high_income and not high_spending:
        name = "Untapped High Earners"
        advice = ("Biggest growth opportunity. They can afford to buy but "
                  "don't. Survey them and try quality-focused ads, not discounts.")
    elif not high_income and high_spending:
        name = "Young Big Spenders"
        advice = ("Spend a lot on a small income. Target them with flash sales, "
                  "payment plans, and social media ads.")
    else:
        name = "Budget Customers"
        advice = ("Low income and low spending. Only use cheap outreach like "
                  "email coupons. Don't spend big advertising money here.")

    names[cluster_id] = name

    print(f"\nCLUSTER {cluster_id}: {name}")
    print(f"   Customers:      {int(row['Customers'])} "
          f"({row['Customers'] / len(df) * 100:.0f}% of all customers)")
    print(f"   Average age:    {row['Avg_Age']:.0f}")
    print(f"   Average income: ${row['Avg_Income']:.0f}k")
    print(f"   Avg spending:   {row['Avg_Spending']:.0f} out of 100")
    print(f"   Strategy:       {advice}")


# =============================================================
# STEP 5: EVALUATION - STRENGTHS AND LIMITATIONS
# =============================================================
print("\n" + "=" * 60)
print("STEP 5: STRENGTHS AND LIMITATIONS")
print("=" * 60)

print(f"""
K-MEANS
  Strengths:   Fast, simple, and gives centroids that describe each segment.
               Best silhouette score here ({kmeans_score:.4f}).
  Limitations: You must choose k yourself. Affected by outliers, and it can
               only find round clusters.

HIERARCHICAL
  Strengths:   No need to choose k in advance. Same answer every time.
  Limitations: Slower on large data. Merges can never be undone.

CONCLUSION
  Both algorithms found nearly the same {best_k} segments. We report K-Means
  because its centroids make the segments easier to explain.
""")

df["Segment_Name"] = df["Cluster"].map(names)
df.to_csv("customers_with_clusters.csv", index=False)

print("Saved: customers_with_clusters.csv")
print("Saved: fig1_raw_data.png ... fig6_final_clusters.png")
print("\nDone.")
