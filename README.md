# Titanic — Exploratory Data Analysis (EDA)

Internship Project: **Exploratory Data Analysis**
Analyzes the Titanic passenger dataset to uncover patterns and trends behind passenger
survival, using statistical summaries, visualizations, and correlation analysis.

## 📁 Project Structure

```
eda-project/
├── data/
│   └── titanic.csv              # Raw dataset (891 passengers)
├── notebooks/
│   └── EDA_Titanic.ipynb        # Main analysis notebook (run this)
├── images/                      # Saved plots (auto-generated)
├── report/
│   └── EDA_Report.md            # Written report with structured insights
├── run_analysis.py              # Standalone script version of the analysis
├── requirements.txt
└── README.md
```

## 📊 Dataset

The [Titanic dataset](https://www.kaggle.com/c/titanic) contains demographic and
ticket information for 891 passengers aboard the RMS Titanic (1912), including:

| Column | Description |
|---|---|
| `Survived` | Target variable (0 = Died, 1 = Survived) |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex`, `Age` | Demographics |
| `SibSp`, `Parch` | Siblings/spouses and parents/children aboard |
| `Fare`, `Embarked` | Ticket fare and port of embarkation |

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd eda-project
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the analysis

**Option A — Jupyter Notebook (recommended for submission/review):**
```bash
jupyter notebook notebooks/EDA_Titanic.ipynb
```
Then click **Run All** (Cell → Run All) to reproduce every output and chart.

**Option B — Standalone script** (generates all plots into `images/`):
```bash
python run_analysis.py
```

## 🔍 Key Insights

- Only **~38%** of passengers survived overall.
- **Sex** was the strongest predictor: women survived at ~74% vs. ~19% for men.
- **Passenger class** strongly affected survival: 1st class ≈ 63% vs. 3rd class ≈ 24%.
- **Fare** correlates positively with survival (a proxy for class).
- **Children** had a survival advantage over adults.
- Passengers with a **small family (2–4 members)** survived more than solo travelers
  or large families.

Full write-up with charts: see [`report/EDA_Report.md`](report/EDA_Report.md).

## 🛠️ Tools Used

Python · pandas · NumPy · Matplotlib · Seaborn · Jupyter Notebook

## 📌 Outcome

This project demonstrates analytical thinking and data exploration skills: cleaning
raw data, summarizing it statistically, visualizing distributions and relationships,
and translating findings into clear, structured insights.
