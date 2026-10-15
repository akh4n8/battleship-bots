import pandas as pd

print('Loading 100k dataset...')
# There are no headers if not written, but wait, if there are headers, they match the export.
df = pd.read_csv('data/Battleship_Synthetic_100k.csv')
# The easiest way to get shots to win is just counting the number of Offense rows per game
offense_shots = df[df['Grid'] == 'Offense']
shots_per_game = offense_shots.groupby('GameID').size()
print(f'Average shots to win for Sherlock over {len(shots_per_game)} games: {shots_per_game.mean():.2f}')
print(f'Median shots to win: {shots_per_game.median():.2f}')
print(f'Min shots: {shots_per_game.min()}')
print(f'Max shots: {shots_per_game.max()}')
