import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

def get_dataset_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'IPL_cleaned_2024.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return csv_path

def save_plot(filename):
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{filename}.png"))
    plt.clf()
    print(f"✅ Saved: {filename}.png in 'visualizations/'")

def create_eda_visualizations():
    dataset_path = get_dataset_path()
    df = pd.read_csv(dataset_path)
    print("📄 Dataset Loaded. Shape:", df.shape)

    # Data Cleaning
    df['runningScore'] = df['runningScore'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
    df['runs'] = df['runningScore'].apply(lambda x: x.get('runs') if isinstance(x, dict) else np.nan)
    df['wickets'] = df['runningScore'].apply(lambda x: x.get('wickets') if isinstance(x, dict) else np.nan)
    df['strikeRate'] = pd.to_numeric(df['strikeRate'], errors='coerce')
    df['ballsFaced'] = pd.to_numeric(df['ballsFaced'], errors='coerce')
    df['runningOver'] = df['runningOver'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['runningOver'] = pd.to_numeric(df['runningOver'], errors='coerce')

    # 📊 Univariate Analysis: Histograms
    df[['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate', 'wickets']].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Histograms of Numerical Features")
    save_plot("histograms_numeric_features")

    # 📈 Bivariate: Runs vs Balls Faced
    sns.scatterplot(x='ballsFaced', y='runs', data=df)
    plt.title("Runs vs Balls Faced")
    save_plot("runs_vs_balls")

    # 📌 Groupby: Average Strike Rate by Team
    team_sr = df.groupby('home_team')['strikeRate'].mean().sort_values(ascending=False)
    team_sr.plot(kind='barh', color='teal')
    plt.title("Average Strike Rate by Team")
    plt.xlabel("Strike Rate")
    save_plot("team_avg_strike_rate")

    # 🎯 Strike Rate Distribution of Top 10 Run Scorers
    top_scorers = df.groupby('name')['runs'].sum().sort_values(ascending=False).head(10).index
    sns.boxplot(data=df[df['name'].isin(top_scorers)], x='name', y='strikeRate')
    plt.xticks(rotation=45)
    plt.title("Strike Rate Distribution of Top 10 Run Scorers")
    save_plot("strike_rate_dist_top_10_run_scorers") 

    # 📦 Top 10 Four Hitters
    top_fours = df.groupby('name')['fours'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_fours.values, y=top_fours.index, palette='crest')
    plt.title("Top 10 Four Hitters")
    plt.xlabel("Total Fours")
    save_plot("top_10_four_hitters")

    # 🎯 Top 10 Six Hitters
    top_sixes = df.groupby('name')['sixes'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_sixes.values, y=top_sixes.index, color='mediumseagreen')
    plt.title("Top 10 Six Hitters")
    plt.xlabel("Total Sixes")
    save_plot("top_10_six_hitters")

    # ⚡ Powerplay: Average Runs (0-6 Overs)
    powerplay_df = df[df['runningOver'] <= 6.0]
    pp_avg_runs = powerplay_df.groupby('current_innings')['runs'].mean().sort_values(ascending=False)
    pp_avg_runs.plot(kind='barh', color='orange')
    plt.title("Powerplay Average Runs by Team (0-6 Overs)")
    plt.xlabel("Average Runs")
    save_plot("powerplay_avg_runs")

    # ⚡ Powerplay: Average Wickets Lost (0-6 Overs)
    pp_avg_wickets = powerplay_df.groupby('current_innings')['wickets'].mean().sort_values(ascending=False)
    pp_avg_wickets.plot(kind='barh', color='green')
    plt.title("Powerplay Average Wickets Lost by Team (0-6 Overs)")
    plt.xlabel("Average Wickets Lost")
    save_plot("powerplay_avg_wickets")

    # 🧬 Correlation Heatmap
    corr = df[['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate', 'wickets']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap")

    # 📈 Pairplot
    sns.pairplot(df[['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate']].dropna(), diag_kind='kde')
    plt.suptitle("Pairplot of Batting Stats", y=1.02)
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    plt.savefig(os.path.join(viz_dir, "pairplot_batting_features.png"))
    plt.clf()
    print("✅ Saved: pairplot_batting_features.png in 'visualizations/'")

    print("\n🎯 All EDA plots saved successfully in 'visualizations/' folder.")

if __name__ == '__main__':
    try:
        create_eda_visualizations()
    except Exception as e:
        print(f"❌ Error: {e}")