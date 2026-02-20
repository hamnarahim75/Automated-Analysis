# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "httpx",
#   "tenacity"
# ]
# ///

import sys
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


# ----------------------------
# LOAD DATA
# ----------------------------
def load_data(filepath):
    return pd.read_csv(filepath)


# ----------------------------
# GENERIC ANALYSIS
# ----------------------------
def basic_analysis(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe(include="number").to_dict()
    }


def detect_outliers(df):
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {}
    z_scores = (numeric - numeric.mean()) / numeric.std()
    return (np.abs(z_scores) > 3).sum().to_dict()


def top_categories(df):
    cat_cols = df.select_dtypes(include="object")
    result = {}
    for col in cat_cols:
        result[col] = df[col].value_counts().head(5).to_dict()
    return result


# ----------------------------
# VISUALIZATION
# ----------------------------
def plot_correlation(df):
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None

    corr = numeric.corr()
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation.png")
    plt.close()
    return "correlation.png"


def plot_distribution(df):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return None

    col = numeric_cols[0]
    plt.figure(figsize=(6, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig("distribution.png")
    plt.close()
    return "distribution.png"


# ----------------------------
# LLM CALL
# ----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(prompt):
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        raise ValueError("AIPROXY_TOKEN not set.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
        ]
    }

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ----------------------------
# MAIN
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = load_data(filename)

    analysis = basic_analysis(df)
    outliers = detect_outliers(df)
    categories = top_categories(df)

    img1 = plot_correlation(df)
    img2 = plot_distribution(df)
    prompt = f"""
Dataset Name: {filename}
Shape: {analysis['shape']}
Columns: {analysis['columns']}

Missing Values:
{json.dumps(analysis['missing_values'], indent=2)}

Numeric Summary:
{json.dumps(analysis['numeric_summary'], indent=2)}

Outliers:
{json.dumps(outliers, indent=2)}

Top Categories:
{json.dumps(categories, indent=2)}
"""


    #----
    # report = call_llm(prompt)
    #----
    report = "Test report generated successfully."

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n## Visualizations\n\n")
        if img1:
            f.write(f"![Correlation Heatmap]({img1})\n\n")
        if img2:
            f.write(f"![Distribution]({img2})\n")

    print("Analysis complete. README.md and PNG files created.")


if __name__ == "__main__":
    main()
