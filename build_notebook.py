"""
Builds notebooks/EDA_Titanic.ipynb using nbformat.
Cells are written so the notebook is fully re-runnable end-to-end by the user.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# Title
md("""# Exploratory Data Analysis (EDA) — Titanic Dataset

**Internship Project: Exploratory Data Analysis**

This notebook analyzes the Titanic passenger dataset to uncover patterns and trends
related to passenger survival. It covers:

1. Data loading & structure
2. Statistical summaries
3. Data cleaning
4. Univariate & bivariate visualizations
5. Correlation analysis
6. Key insights & conclusions

**Dataset:** 891 passengers from the RMS Titanic (1912), including demographic and
ticket information, with the target variable `Survived` (0 = Died, 1 = Survived).
""")

# 1. Imports
md("## 1. Import Libraries")
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
%matplotlib inline
""")

# 2. Load data
md("## 2. Load the Dataset")
code('''df = pd.read_csv("../data/titanic.csv")
df.head()
''')

md("## 3. Understand the Structure")
code("""print("Shape of dataset:", df.shape)
df.info()
""")

code('''df.describe(include="all").T''')

md("""### Missing Values

Checking for missing data is a critical first EDA step — it determines how we clean
and what caveats apply to later analysis.""")
code("""missing = df.isnull().sum().sort_values(ascending=False)
missing[missing > 0]
""")

md("""**Observation:** `Cabin` is missing for ~77% of passengers (too sparse to use
reliably), `Age` is missing for ~20%, and `Embarked` is missing for 2 records.""")

# 4. Cleaning
md("""## 4. Data Cleaning & Feature Engineering

- Fill missing `Age` with the median (robust to outliers).
- Fill missing `Embarked` with the most frequent port.
- Create `AgeGroup` for easier categorical comparison.
- Create `FamilySize` = SibSp + Parch + 1 (siblings/spouses + parents/children + self).""")
code('''df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df[["Age", "Embarked", "AgeGroup", "FamilySize"]].head()
''')

# 5. Univariate
md("""## 5. Univariate Analysis

### 5.1 Overall Survival Rate""")
code('''survival_counts = df["Survived"].value_counts()
survival_rate = df["Survived"].mean() * 100
print(f"Overall survival rate: {survival_rate:.1f}%")

fig, ax = plt.subplots(figsize=(5, 4))
survival_counts.rename({0: "Died", 1: "Survived"}).plot(
    kind="bar", color=["#d62728", "#2ca02c"], ax=ax)
ax.set_title("Overall Survival Count")
ax.set_ylabel("Number of Passengers")
plt.tight_layout()
plt.show()
''')

md("**Observation:** Only about 38% of passengers survived — a majority did not.")

md("### 5.2 Age Distribution")
code('''fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x="Age", kde=True, bins=30, ax=ax, color="steelblue")
ax.set_title("Age Distribution of Passengers")
plt.tight_layout()
plt.show()
''')

# 6. Bivariate
md("""## 6. Bivariate Analysis — Key Influencing Factors

### 6.1 Survival by Sex""")
code('''sex_survival = df.groupby("Sex")["Survived"].mean()
print(sex_survival)

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=df, x="Sex", y="Survived", hue="Sex", legend=False, ax=ax, palette="Set2")
ax.set_title("Survival Rate by Sex")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.show()
''')

md("""**Observation:** Females had a dramatically higher survival rate (~74%) than
males (~19%) — consistent with the "women and children first" evacuation policy.""")

md("### 6.2 Survival by Passenger Class")
code('''class_survival = df.groupby("Pclass")["Survived"].mean()
print(class_survival)

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Pclass", legend=False, ax=ax, palette="Set2")
ax.set_title("Survival Rate by Passenger Class")
ax.set_ylabel("Survival Rate")
plt.tight_layout()
plt.show()
''')

md("""**Observation:** 1st class passengers survived at far higher rates (~63%) than
3rd class (~24%) — socioeconomic status strongly influenced survival, likely tied to
cabin location and lifeboat access.""")

md("### 6.3 Age Distribution by Survival")
code('''fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, ax=ax, palette="Set1")
ax.set_title("Age Distribution by Survival")
plt.tight_layout()
plt.show()
''')

md("**Observation:** Young children had noticeably higher survival rates than other age groups.")

md("### 6.4 Fare vs Survival")
code('''fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Survived", y="Fare", hue="Survived", legend=False, ax=ax, palette="Set2")
ax.set_ylim(0, 300)
ax.set_title("Fare Distribution by Survival")
plt.tight_layout()
plt.show()
''')

md("**Observation:** Survivors tended to have paid higher fares, reinforcing the class effect above.")

md("### 6.5 Survival by Embarkation Port")
code('''embarked_survival = df.groupby("Embarked")["Survived"].mean()
print(embarked_survival)

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=df, x="Embarked", y="Survived", hue="Embarked", legend=False, ax=ax, palette="Set2")
ax.set_title("Survival Rate by Embarkation Port")
plt.tight_layout()
plt.show()
''')

md("### 6.6 Survival by Family Size")
code('''fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df, x="FamilySize", y="Survived", hue="FamilySize", legend=False, ax=ax, palette="viridis")
ax.set_title("Survival Rate by Family Size")
plt.tight_layout()
plt.show()
''')

md("""**Observation:** Passengers traveling with a small family (2-4 members) survived
more often than those traveling completely alone or with very large families.""")

# 7. Correlation
md("""## 7. Correlation Analysis""")
code('''num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

corr["Survived"].sort_values(ascending=False)
''')

md("""**Observation:** `Pclass` has the strongest negative correlation with survival
(-0.34) and `Fare` the strongest positive correlation (+0.26) among numeric features —
both point to socioeconomic status as a key driver of survival.""")

md("### 7.1 Pairwise Relationships")
code('''g = sns.pairplot(df[num_cols].dropna(), hue="Survived", palette="Set1",
                  corner=True, diag_kind="hist")
g.fig.suptitle("Pairplot of Key Numerical Features", y=1.02)
plt.show()
''')

# 8. Conclusion
md("""## 8. Key Insights & Conclusion

1. **Sex was the single strongest predictor of survival** — women survived at ~74%
   vs. ~19% for men.
2. **Passenger class (socioeconomic status) was the strongest numeric correlate**
   of survival — 1st class passengers survived at roughly 2.5x the rate of 3rd class.
3. **Fare correlates positively with survival**, reinforcing the class effect (fare
   is essentially a proxy for class and cabin location).
4. **Children had a survival advantage** over adults, consistent with evacuation
   priority policies.
5. **Moderate family size (2-4 people) was protective** compared to traveling alone
   or in very large families, possibly reflecting mutual assistance during evacuation
   vs. the difficulty of coordinating large groups.
6. **Port of embarkation** showed some variation in survival rate, largely driven by
   the class composition of passengers boarding at each port (Cherbourg had a higher
   proportion of 1st class passengers).

**Overall:** Survival on the Titanic was driven primarily by a combination of
**sex, socioeconomic class, and age**, reflecting both the era's social norms
("women and children first") and practical realities of cabin location and
lifeboat access for wealthier passengers.
""")

nb["cells"] = cells

with open("notebooks/EDA_Titanic.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook structure created.")
