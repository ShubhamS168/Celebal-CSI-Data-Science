import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast

def get_dataset_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'IPL_cleaned_2024.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return csv_path

def create_visualizations():
    dataset_path = get_dataset_path()
    df = pd.read_csv(dataset_path)
    print("Dataset loaded. Columns are:", df.columns.tolist())

    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    def save_plot(filename):
        plt.savefig(os.path.join(viz_dir, filename))
        plt.clf()
        print(f"Saved: {filename} in 'visualizations/'")

    # Data preprocessing
    df['runs'] = df['runningScore'].apply(lambda x: ast.literal_eval(x).get('runs') if pd.notnull(x) else None)
    df['wickets'] = df['runningScore'].apply(lambda x: ast.literal_eval(x).get('wickets') if pd.notnull(x) and 'wickets' in ast.literal_eval(x) else None)
    df['strikeRate'] = pd.to_numeric(df['strikeRate'], errors='coerce')
    df['ballsFaced'] = pd.to_numeric(df['ballsFaced'], errors='coerce')
    df['runningOver'] = df['runningOver'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['runningOver'] = pd.to_numeric(df['runningOver'], errors='coerce')

    # 🔹 Top 10 players with the most sixes in IPL 2024
    top_run_df = df.groupby('name')['sixes'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_run_df.values, y=top_run_df.index, palette='YlOrBr', legend=False)
    plt.title('Top 10 Six hitters in IPL 2024')
    plt.xlabel('Total Sixes')
    plt.ylabel('Batsmen')
    plt.tight_layout()
    save_plot('top_10_Six_hitter.png')

    # 🔹 Top 10 players with the highest average strike rate across the tournament
    top_strike = df.groupby('name')['strikeRate'].mean().dropna().sort_values(ascending=False).head(10)
    plt.figure(figsize=(11, 6))
    sns.barplot(x=top_strike.values, y=top_strike.index, palette='Blues_d', legend=False)
    plt.title('Top 10 Players with Highest Strike Rate')
    plt.xlabel('Average Strike Rate')
    save_plot('top_strike_rate_players.png')

    # 🔹 Top 10 players with strike rate among those who scored more than 200 runs
    top_strike = df.groupby('name').agg({'strikeRate': 'mean', 'runs': 'sum'})
    top_strike = top_strike[top_strike['runs'] > 200].sort_values('strikeRate', ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_strike['strikeRate'].values, y=top_strike.index, palette='Blues_d', legend=False)
    plt.title('Top Strike Rate Players (Runs > 200)')
    plt.xlabel('Average Strike Rate')
    plt.ylabel('Batsmen')
    plt.tight_layout()
    save_plot('top_strike_rate_players_Runs_above_200.png')

    # 🔹 Strike rate comparison of top 10 Royal Challengers Bangalore (RCB) batsmen
    same_team_strike = df[df['current_innings'] == 'RCB'].groupby('name')['strikeRate'].mean().reset_index()
    same_team_strike = same_team_strike.sort_values('strikeRate', ascending=False).head(10)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=same_team_strike, x='strikeRate', y='name', palette='cool', legend=False)
    plt.title('Strike Rate of RCB Players')
    plt.xlabel('Strike Rate')
    plt.ylabel('RCB Batsmen')
    plt.tight_layout()
    save_plot('rcb_players_strike_rate.png')

    # 🔹 Average strike rate for each team to compare overall team performance
    team_strike = df.groupby('home_team')['strikeRate'].mean().reset_index().sort_values('strikeRate', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=team_strike, x='strikeRate', y='home_team', palette='mako', legend=False)
    plt.title('Comparison of Average Team Strike Rates')
    plt.xlabel('Average Strike Rate')
    plt.ylabel('Team')
    save_plot('team_strike_rate_comparison.png')

    # 🔹 Top 10 players with the highest average number of boundaries (fours and sixes)
    player_boundaries = df.groupby('name')[['fours', 'sixes']].mean().sort_values(by='fours', ascending=False).head(10)
    player_boundaries.plot(kind='barh', figsize=(10, 6), colormap='Set2')
    plt.title('Average Fours and Sixes per Player (Top 10)')
    plt.xlabel('Average Count')
    save_plot('avg_fours_sixes_per_player.png')

    # 🔹 Average runs scored across different venues to assess pitch/batting conditions
    plt.figure(figsize=(16, 10))
    sns.barplot(data=df, x='runs', y='venue',  hue='venue', estimator='mean', errorbar=None)
    plt.title('Average Runs by Venue', fontsize=18)
    plt.xlabel('Average Runs', fontsize=14)
    plt.ylabel('Venue', fontsize=14)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    save_plot('avg_runs_by_venue.png')

    # 🔹 Heatmap showing correlation between numeric batting features
    plt.figure(figsize=(10, 9))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    save_plot('correlation_heatmap.png')

    # 🔹 Pairplot showing relationships between runs, balls faced, boundaries and strike rate
    key_feats = ['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate']
    sns.pairplot(df[key_feats].dropna(), diag_kind='kde', plot_kws={'alpha': 0.7})
    plt.suptitle('Pairplot of Key Batting Features', y=1.02, fontsize=16)
    plt.savefig(os.path.join(viz_dir, 'pairplot_batting_features.png'))
    plt.clf()
    print("Saved: pairplot_batting_features.png in 'visualizations/'")

    # 🔹 Top 3 six-hitters from each team based on total sixes
    top_batsmen_teamwise = df.groupby(['current_innings', 'name'])['sixes'].sum().reset_index()
    top_batsmen_teamwise = top_batsmen_teamwise.sort_values(['current_innings', 'sixes'], ascending=[True, False])
    top3_per_team = top_batsmen_teamwise.groupby('current_innings').head(3)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top3_per_team, x='sixes', y='name', hue='current_innings', dodge=False)
    plt.title('Top 3 Six hitter Per Team by Runs')
    plt.xlabel('Total Sixes')
    plt.ylabel('Six hitter')
    plt.tight_layout()
    save_plot('top_Six_hitters_teamwise.png')

    # 🔹 Total number of boundaries (fours and sixes) scored by each team
    team_boundaries = df.groupby('current_innings')[['fours', 'sixes']].sum().reset_index()
    team_boundaries.plot(x='current_innings', kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')
    plt.title('Total Fours and Sixes by Team')
    plt.ylabel('Total Boundaries')
    plt.xlabel('Team')
    plt.xticks(rotation=45)
    save_plot('team_fours_sixes.png')

    # 🔹 Teams that lost the most wickets during the powerplay (first 6 overs)
    powerplay_df = df[df['runningOver'] <= 6.0]
    wickets_in_powerplay = powerplay_df.groupby('current_innings')['wickets'].max().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=wickets_in_powerplay.values, y=wickets_in_powerplay.index, palette='Reds', legend=False)
    plt.title('Most Wickets Lost in Powerplay (upto 6.0 Overs)')
    plt.xlabel('Max Wickets Lost')
    plt.ylabel('Team')
    save_plot('wickets_lost_powerplay.png')

    # 🔹 Teams that scored the most runs during the powerplay (first 6 overs)
    runs_in_powerplay = powerplay_df.groupby('current_innings')['runs'].max().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=runs_in_powerplay.values, y=runs_in_powerplay.index, palette='Greens', legend=False)
    plt.title('Most Runs Scored in Powerplay (upto 6.0 Overs)')
    plt.xlabel('Max Runs Scored')
    plt.ylabel('Team')
    save_plot('runs_scored_in_powerplay_teamwise.png')

    print("\n✅ All visualizations saved in 'visualizations/' folder.")

if __name__ == '__main__':
    create_visualizations()
