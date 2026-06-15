# VedGrow_MS_001
Exploratory Data Analysis (EDA) on the Titanic dataset to uncover demographic and socioeconomic survival patterns for the VedGrow Data Science Internship.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load Data & Check Basic structures
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("--- Dataset Shape ---")
print(df.shape)
print("\n--- Data Types & Info ---")
print(df.info())
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 2. Survival Rate Analysis by Key Demographics
print("\n--- Survival Rate by Gender ---")
print(df.groupby("Sex")["Survived"].mean() * 100)

print("\n--- Survival Rate by Passenger Class ---")
print(df.groupby("Pclass")["Survived"].mean() * 100)

# Create structural age groups to uncover deep patterns
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
)
print("\n--- Survival Rate by Age Group ---")
print(df.groupby("AgeGroup", observed=False)["Survived"].mean() * 100)

# 3. Comprehensive Visualizations Suite
plt.figure(figsize=(14, 10))

# Plot A: Correlation Heatmap of Numerical Data
plt.subplot(2, 2, 1)
numeric_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix Heatmap")

# Plot B: Distribution of Passenger Ages by Survival State
plt.subplot(2, 2, 2)
sns.histplot(data=df, x="Age", hue="Survived", kde=True, multiple="stack", palette="muted")
plt.title("Age Distribution by Survival")

# Plot C: Distribution of Fare metrics
plt.subplot(2, 2, 3)
sns.histplot(data=df, x="Fare", hue="Survived", kde=True, multiple="stack", palette="magma")
plt.xlim(0, 250)  # Capping visibility to filter out outliers
plt.title("Fare Distribution Matrix")

# Plot D: Survival breakdown across Passenger Cabin Classes
plt.subplot(2, 2, 4)
sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set2")
plt.title("Survival Count across Classes")

plt.tight_layout()
plt.savefig("titanic_eda_summary.png", dpi=300)
plt.show()
