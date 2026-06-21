"""
Exploratory Data Analysis - Titanic Dataset
Generates all statistical summaries and visualizations used in the notebook/report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/titanic.csv")

# ---------------------------------------------------------------
# 1. Basic structure
# ---------------------------------------------------------------
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe(include="all"))

# ---------------------------------------------------------------
# 2. Cleaning (lightweight, for visualization purposes only)
# ---------------------------------------------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# ---------------------------------------------------------------
# 3. Survival rate overview
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
df["Survived"].value_counts().rename({0: "Died", 1: "Survived"}).plot(
    kind="bar", color=["#d62728", "#2ca02c"], ax=ax)
ax.set_title("Overall Survival Count")
ax.set_xlabel("")
ax.set_ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("images/01_survival_overview.png")
plt.close()

# ---------------------------------------------------------------
# 4. Survival by Sex
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=df, x="Sex", y="Survived", ax=ax, palette="Set2")
ax.set_title("Survival Rate by Sex")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("images/02_survival_by_sex.png")
plt.close()

# ---------------------------------------------------------------
# 5. Survival by Passenger Class
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=df, x="Pclass", y="Survived", ax=ax, palette="Set2")
ax.set_title("Survival Rate by Passenger Class")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("images/03_survival_by_class.png")
plt.close()

# ---------------------------------------------------------------
# 6. Age distribution
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, ax=ax, palette="Set1")
ax.set_title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("images/04_age_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 7. Fare distribution
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Survived", y="Fare", ax=ax, palette="Set2")
ax.set_title("Fare Distribution by Survival")
ax.set_ylim(0, 300)
plt.tight_layout()
plt.savefig("images/05_fare_by_survival.png")
plt.close()

# ---------------------------------------------------------------
# 8. Survival by Embarkation Point
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=df, x="Embarked", y="Survived", ax=ax, palette="Set2")
ax.set_title("Survival Rate by Embarkation Port")
plt.tight_layout()
plt.savefig("images/06_survival_by_embarked.png")
plt.close()

# ---------------------------------------------------------------
# 9. Family size effect
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df, x="FamilySize", y="Survived", ax=ax, palette="viridis")
ax.set_title("Survival Rate by Family Size")
plt.tight_layout()
plt.savefig("images/07_survival_by_familysize.png")
plt.close()

# ---------------------------------------------------------------
# 10. Correlation heatmap
# ---------------------------------------------------------------
num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("images/08_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 11. Pairwise relationships
# ---------------------------------------------------------------
g = sns.pairplot(df[num_cols].dropna(), hue="Survived",
                  palette="Set1", corner=True, diag_kind="hist")
g.fig.suptitle("Pairplot of Key Numerical Features", y=1.02)
g.savefig("images/09_pairplot.png")
plt.close()

print("\nCorrelation with Survived:\n", corr["Survived"].sort_values(ascending=False))
print("\nAll plots saved to images/. Analysis complete.")
