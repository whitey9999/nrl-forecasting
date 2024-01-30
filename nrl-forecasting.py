import pandas as pd
import numpy as np
import datetime
import math
from scipy.stats import norm


def clean_nrl_scores(score_data):
    df_scores = pd.read_csv(score_data)
    df_scores = df_scores.replace('Dolphins', 'Redcliffe Dolphins')
    df_scores = df_scores.replace('Sea Eagles', 'Manly Sea Eagles')

    if len(df_scores['Home Team'].unique()) == 17:
        df_scores.to_csv('nrl_scores_2023_clean.csv', index=False)
    else:
        print('Error with Team Names')
        print(df_scores['Home Team'].unique())

    try:
        df_scores['Date'] = pd.to_datetime(df_scores['Date'], format='%d/%m/%Y')
    except:
        df_scores['Date'] = pd.to_datetime(df_scores['Date'], format='%Y/%m/%d')

    return df_scores


def enter_new_scores(df_scores):
    for index, row in df_scores.iterrows():
        if np.isnan(row['Home Score']) and row['Date'] <= datetime.datetime.now():
            home_team = row['Home Team']
            away_team = row['Away Team']
            match_date = row['Date']
            print(f'{match_date}: {home_team} vs {away_team} ')
            home_score = input(f'{home_team}: ')
            away_score = input(f'{away_team}: ')
            print(f'{home_team} {home_score} vs {away_team} {away_score}')
            if home_score != '':
                df_scores.at[index, 'Home Score'] = int(home_score)
                df_scores.at[index, 'Away Score'] = int(away_score)

    return df_scores


def team_statistics(df_games_played):
    df_games_played['Home Win'] = 1
    df_games_played.loc[df_games_played['Away Score'] > df_games_played['Home Score'], 'Home Win'] = 0
    df_games_played.loc[df_games_played['Away Score'] == df_games_played['Home Score'], 'Home Win'] = 0.5
    df_games_played['Away Win'] = 1 - df_games_played['Home Win']

    # Offence and Defence Averages by Home and Away
    home_averages = df_games_played.groupby(['Home Team'])['Home Score'].mean()
    away_averages = df_games_played.groupby(['Away Team'])['Away Score'].mean()
    home_averages_ag = df_games_played.groupby(['Home Team'])['Away Score'].mean()
    away_averages_ag = df_games_played.groupby(['Away Team'])['Home Score'].mean()

    # Home and Away Count
    home_count = df_games_played.groupby(['Home Team'])['Home Team'].count()
    away_count = df_games_played.groupby(['Away Team'])['Away Team'].count()

    wins = df_games_played.groupby(['Home Team'])['Home Win'].sum() + \
           df_games_played.groupby(['Away Team'])['Away Win'].sum()

    team_table = pd.concat([home_averages, home_averages_ag, home_count,
                            away_averages, away_averages_ag, away_count, wins], axis=1)
    team_table.columns = ['home_score', 'home_score_ag', 'home_count',
                          'away_score', 'away_score_ag', 'away_count', 'wins']

    team_table['points_for'] = team_table.apply(lambda row: row['home_score'] * row['home_count'] +
                                                            row['away_score'] * row['away_count'], axis=1)

    team_table['points_against'] = team_table.apply(lambda row: row['home_score_ag'] * row['home_count'] +
                                                                row['away_score_ag'] * row['away_count'], axis=1)

    team_table['average_for'] = team_table.apply(
        lambda row: row['points_for'] / (row['home_count'] + row['away_count']), axis=1)
    team_table['average_against'] = team_table.apply(
        lambda row: row['points_against'] / (row['home_count'] + row['away_count']), axis=1)

    return team_table


def error_measure(model_name, error_series):
    rsme = np.sqrt((error_series ** 2).sum() / error_series.shape[0])

    error_within_six = error_series.loc[(error_series < 6) & (error_series > -6)].count() / error_series.shape[0]

    error_within_twelve = error_series.loc[(error_series < 12) & (error_series > -12)].count() / error_series.shape[0]

    print(f'\n{model_name} Model Error measurements: \nRSME: {rsme:.4f}')
    print(f'Within 6 pts: {error_within_six:.2%}\nWithin 12 pts:{error_within_twelve:.2%}')

    return 0


def dvoa_model(df_games_played, df_team_stats):

    df_dvoa = df_games_played.copy()

    # Convert average scores to dictionaries
    pts_for_avg = dict(zip(df_team_stats.index, df_team_stats['average_for']))
    pts_ag_avg = dict(zip(df_team_stats.index, df_team_stats['average_against']))

    # Map the dictionary averages to the dataframe
    df_dvoa['home_avg_for'] = df_dvoa['Home Team'].map(pts_for_avg)
    df_dvoa['away_avg_for'] = df_dvoa['Away Team'].map(pts_for_avg)
    df_dvoa['home_avg_ag'] = df_dvoa['Home Team'].map(pts_ag_avg)
    df_dvoa['away_avg_ag'] = df_dvoa['Away Team'].map(pts_ag_avg)

    # Record the difference between the opponent average for each game
    df_dvoa['home_dif_for'] = df_dvoa.apply(lambda row: row['Home Score'] - row['away_avg_ag'], axis=1)
    df_dvoa['home_dif_against'] = df_dvoa.apply(lambda row: row['Away Score'] - row['away_avg_for'], axis=1)
    df_dvoa['away_dif_for'] = df_dvoa.apply(lambda row: row['Away Score'] - row['home_avg_ag'], axis=1)
    df_dvoa['away_dif_against'] = df_dvoa.apply(lambda row: row['Home Score'] - row['home_avg_for'], axis=1)

    # Adjust outliers to within 3 standard deviations to reduce weighting
    for column in df_dvoa[['home_dif_for', 'home_dif_against', 'away_dif_for', 'away_dif_against']]:
        threshold = df_dvoa[column].std() * 3
        df_dvoa[column] = df_dvoa[column].apply(lambda x: x if x < threshold else threshold)
        df_dvoa[column] = df_dvoa[column].apply(lambda x: x if x > -threshold else -threshold)

    # Calculate the total difference vs opponent average
    off_dif = df_dvoa.groupby(['Home Team'])['home_dif_for'].sum() + \
              df_dvoa.groupby(['Away Team'])['away_dif_for'].sum()
    def_dif = df_dvoa.groupby(['Home Team'])['home_dif_against'].sum() + \
              df_dvoa.groupby(['Away Team'])['away_dif_against'].sum()

    # Convert to dictionary
    off_dif_dict = dict(off_dif.round(2))
    def_dif_dict = dict(def_dif.round(2))

    # Calculate the average and map the team statistics dataframe
    df_team_stats['dvoa_off'] = df_team_stats.index.map(off_dif_dict) / (
                df_team_stats['home_count'] + df_team_stats['away_count'])
    df_team_stats['dvoa_def'] = df_team_stats.index.map(def_dif_dict) / (
                df_team_stats['home_count'] + df_team_stats['away_count'])
    df_team_stats = df_team_stats.round(2)

    # Dictionary for DVOA ratings
    pts_for_dvoa = dict(zip(df_team_stats.index, df_team_stats['dvoa_off']))
    pts_ag_dvoa = dict(zip(df_team_stats.index, df_team_stats['dvoa_def']))

    # Map the DVOA ratings to the game dataframe
    df_dvoa['home_off_dvoa'] = df_dvoa['Home Team'].map(pts_for_dvoa)
    df_dvoa['home_def_dvoa'] = df_dvoa['Home Team'].map(pts_ag_dvoa)
    df_dvoa['away_off_dvoa'] = df_dvoa['Away Team'].map(pts_for_dvoa)
    df_dvoa['away_def_dvoa'] = df_dvoa['Away Team'].map(pts_ag_dvoa)

    # Calculate projections for the matches
    df_dvoa['home_projection'] = df_dvoa.apply(lambda row: (row['home_avg_for'] + row['away_def_dvoa'] +
                                                            row['away_avg_ag'] + row['home_off_dvoa']) / 2, axis=1)
    df_dvoa['away_projection'] = df_dvoa.apply(lambda row: (row['away_avg_for'] + row['home_def_dvoa'] +
                                                            row['home_avg_ag'] + row['away_off_dvoa']) / 2, axis=1)

    # Calculate the error on the DVOA projections
    df_dvoa['proj_dif'] = df_dvoa['Home Score'] - df_dvoa['home_projection'] + df_dvoa['Away Score'] - df_dvoa[
        'away_projection']

    error_measure('DVOA', df_dvoa['proj_dif'])

    df_dvoa.to_csv('nrl_2023_dvoa.csv')

    return df_team_stats


def dvohaa_model(df_games_played, df_team_stats):

    df_dvohaa = df_games_played.copy()

    # Dictionary of the team statistics
    pts_home_for_avg = dict(zip(df_team_stats.index, df_team_stats['home_score']))
    pts_home_ag_avg = dict(zip(df_team_stats.index, df_team_stats['home_score_ag']))
    pts_away_for_avg = dict(zip(df_team_stats.index, df_team_stats['away_score']))
    pts_away_ag_avg = dict(zip(df_team_stats.index, df_team_stats['away_score_ag']))

    # Map the team scoring statistics to the model
    df_dvohaa['home_avg_for'] = df_dvohaa['Home Team'].map(pts_home_for_avg)
    df_dvohaa['away_avg_for'] = df_dvohaa['Away Team'].map(pts_away_for_avg)
    df_dvohaa['home_avg_ag'] = df_dvohaa['Home Team'].map(pts_home_ag_avg)
    df_dvohaa['away_avg_ag'] = df_dvohaa['Away Team'].map(pts_away_ag_avg)

    df_dvohaa['home_dif_for'] = df_dvohaa.apply(lambda row: row['Home Score'] - row['away_avg_ag'], axis=1)
    df_dvohaa['home_dif_against'] = df_dvohaa.apply(lambda row: row['Away Score'] - row['away_avg_for'], axis=1)
    df_dvohaa['away_dif_for'] = df_dvohaa.apply(lambda row: row['Away Score'] - row['home_avg_ag'], axis=1)
    df_dvohaa['away_dif_against'] = df_dvohaa.apply(lambda row: row['Home Score'] - row['home_avg_for'], axis=1)

    # Adjust the differences to three standard deviations to avoid blowout weightings

    for column in df_dvohaa[['home_dif_for', 'home_dif_against', 'away_dif_for', 'away_dif_against']]:
        threshold = df_dvohaa[column].std() * 3
        df_dvohaa[column] = df_dvohaa[column].apply(lambda x: x if x < threshold else threshold)
        df_dvohaa[column] = df_dvohaa[column].apply(lambda x: x if x > -threshold else -threshold)

    df_team_stats['dvoa_home_off'] = df_dvohaa.groupby(['Home Team'])['home_dif_for'].mean()
    df_team_stats['dvoa_home_def'] = df_dvohaa.groupby(['Home Team'])['home_dif_against'].mean()
    df_team_stats['dvoa_away_off'] = df_dvohaa.groupby(['Away Team'])['away_dif_for'].mean()
    df_team_stats['dvoa_away_def'] = df_dvohaa.groupby(['Away Team'])['away_dif_against'].mean()

    pts_home_for_dvoa = dict(zip(df_team_stats.index, df_team_stats['dvoa_home_off']))
    pts_home_ag_dvoa = dict(zip(df_team_stats.index, df_team_stats['dvoa_home_def']))
    pts_away_for_dvoa = dict(zip(df_team_stats.index, df_team_stats['dvoa_away_off']))
    pts_away_ag_dvoa = dict(zip(df_team_stats.index, df_team_stats['dvoa_away_def']))

    df_dvohaa['home_dvoha_off'] = df_dvohaa['Home Team'].map(pts_home_for_dvoa)
    df_dvohaa['home_dvoha_def'] = df_dvohaa['Home Team'].map(pts_home_ag_dvoa)
    df_dvohaa['away_dvoha_off'] = df_dvohaa['Away Team'].map(pts_away_for_dvoa)
    df_dvohaa['away_dvoha_def'] = df_dvohaa['Away Team'].map(pts_away_ag_dvoa)

    df_dvohaa['home_projection'] = round(((df_dvohaa['home_avg_for'] + df_dvohaa['away_dvoha_def']) +
                                          (df_dvohaa['away_avg_ag'] + df_dvohaa['home_dvoha_off'])) / 2, 2)

    df_dvohaa['away_projection'] = round(((df_dvohaa['home_avg_ag'] + df_dvohaa['away_dvoha_off']) +
                                         (df_dvohaa['away_avg_for'] + df_dvohaa['home_dvoha_def'])) / 2, 2)

    df_dvohaa['Error'] = df_dvohaa['Home Score'] - df_dvohaa['home_projection'] + \
                        df_dvohaa['Away Score'] - df_dvohaa['away_projection']

    error_measure('DVOHAA', df_dvohaa['Error'])

    df_dvohaa.to_csv('nrl_2023_dvohaa.csv')

    return df_team_stats


def elo_model(df_games_played, df_team_stats):

    df_elo = df_games_played.copy()

    elo_2022 = [1485, 1539, 1426, 1550, 1424, 1447, 1541, 1414, 1409, 1556,
                1571, 1623, 1500, 1565, 1495, 1559, 1396]

    df_team_stats['elo_2022'] = elo_2022
    elo_dict = dict(zip(df_team_stats.index, df_team_stats['elo_2022']))

    for i in range(len(df_elo)):
        df_elo.loc[i, 'home_elo'] = elo_dict[df_elo.loc[i, 'Home Team']]
        df_elo.loc[i, 'away_elo'] = elo_dict[df_elo.loc[i, 'Away Team']]

        df_elo.loc[i, 'proj_margin'] = ((df_elo.loc[i, 'home_elo'] + 37.5) - df_elo.loc[i, 'away_elo']) / 15

        df_elo.loc[i, 'home_chance'] = 1 / (
                    np.power(10, (-(df_elo.loc[i, 'home_elo'] - df_elo.loc[i, 'away_elo'] + 37.5) / 300)) + 1)

        df_elo.loc[i, 'away_chance'] = 1 - df_elo.loc[i, 'home_chance']

        margin = df_elo.loc[i, 'Home Score'] - df_elo.loc[i, 'Away Score']

        # Rnew = ROld + K * MOV * (W-Wexpectation)
        if margin > 0:
            change_elo = (1 - df_elo.loc[i, 'home_chance']) * 30 * \
                         math.log10(1 + margin) * 2.2 / (
                                     (df_elo.loc[i, 'home_elo'] - df_elo.loc[i, 'away_elo'] + 37.5) * 0.001 + 2.2)

        elif margin < 0:
            change_elo = -(1 - df_elo.loc[i, 'away_chance']) * 30 * \
                         math.log10(1 + -margin) * 2.2 / (
                                     (df_elo.loc[i, 'away_elo'] - df_elo.loc[i, 'home_elo'] - 37.5) * 0.001 + 2.2)

        elif margin == 0:
            change_elo = (0.5 - df_elo.loc[i, 'home_chance']) * 30
        else:
            change_elo = 0

        elo_dict[df_elo.loc[i, 'Home Team']] += change_elo
        elo_dict[df_elo.loc[i, 'Away Team']] -= change_elo

    df_elo['Error'] = df_elo['Home Score'] - df_elo['Away Score'] - df_elo['proj_margin']

    error_measure('Elo', df_elo['Error'])

    df_team_stats['elo_2023'] = df_team_stats.index.map(elo_dict)

    return df_team_stats


def update_models(df_scores):
    df_games_played = df_scores[(df_scores['Home Score'] >= 0)]

    df_team_stats = team_statistics(df_games_played)
    df_team_stats = dvoa_model(df_games_played, df_team_stats)

    df_team_stats = dvohaa_model(df_games_played, df_team_stats)

    df_team_stats = elo_model(df_games_played, df_team_stats)

    df_team_stats.to_csv('nrl_2023_team_summary.csv')

    return df_team_stats

def set_projections(df_proj, df_team_stats):

    elo_dict = dict(zip(df_team_stats.index, df_team_stats['elo_2023']))
    df_proj['home_elo'] = df_proj['Home Team'].map(elo_dict)
    df_proj['away_elo'] = df_proj['Away Team'].map(elo_dict)

    df_proj['elo_margin'] = ((df_proj['home_elo'] + 37.5) - df_proj['away_elo']) / 15

    df_proj['elo_home_chance'] = 1 / (np.power(10, (-(df_proj['home_elo'] - df_proj['away_elo'] + 37.5) / 300)) + 1)

    df_proj['elo_away_chance'] = 1.0 - df_proj['elo_home_chance']

    df_proj = df_proj.drop(['home_elo', 'away_elo'], axis=1)

    df_proj['dvoa_home'] = df_proj.apply(lambda row: ((df_team_stats.loc[row['Home Team'], 'average_for'] +
                                                       df_team_stats.loc[row['Home Team'], 'dvoa_off'] +
                                                       df_team_stats.loc[row['Away Team'], 'average_against'] +
                                                       df_team_stats.loc[row['Away Team'], 'dvoa_def']) / 2), axis=1)

    df_proj['dvoa_away'] = df_proj.apply(lambda row: ((df_team_stats.loc[row['Away Team'], 'average_for'] +
                                                       df_team_stats.loc[row['Away Team'], 'dvoa_off'] +
                                                       df_team_stats.loc[row['Home Team'], 'average_against'] +
                                                       df_team_stats.loc[row['Home Team'], 'dvoa_def']) / 2), axis=1)

    df_proj['dvhaa_home'] = df_proj.apply(lambda row: ((df_team_stats.loc[row['Home Team'], 'home_score'] +
                                                        df_team_stats.loc[row['Home Team'], 'dvoa_home_off'] +
                                                        df_team_stats.loc[row['Away Team'], 'away_score_ag'] +
                                                        df_team_stats.loc[row['Away Team'], 'dvoa_away_def']) / 2),
                                          axis=1)

    df_proj['dvhaa_away'] = df_proj.apply(lambda row: ((df_team_stats.loc[row['Away Team'], 'away_score'] +
                                                        df_team_stats.loc[row['Away Team'], 'dvoa_away_off'] +
                                                        df_team_stats.loc[row['Home Team'], 'home_score_ag'] +
                                                        df_team_stats.loc[row['Home Team'], 'dvoa_home_def']) / 2),
                                          axis=1)

    df_proj['dvoa_margin'] = df_proj['dvoa_home'] - df_proj['dvoa_away']
    df_proj['dvhaa_margin'] = df_proj['dvhaa_home'] - df_proj['dvhaa_away']
    df_proj['average_projections'] = (df_proj['dvoa_margin'] + df_proj['dvhaa_margin'] + df_proj['elo_margin']) / 3

    df_proj['home_dvoa_win%'] = norm.cdf(df_proj['dvoa_margin'] / 12)
    df_proj['away_dvoa_win%'] = 1 - df_proj['home_dvoa_win%']

    df_proj['home_dvhaa_win%'] = norm.cdf(df_proj['dvhaa_margin'] / 12)
    df_proj['away_dvhaa_win%'] = 1 - df_proj['home_dvhaa_win%']

    df_proj['home_avg_win%'] = norm.cdf(df_proj['average_projections'] / 12)
    df_proj['away_avg_win%'] = 1 - df_proj['home_avg_win%']

    return df_proj


def simulate_season(df_proj, df_team_stats, run_times):
    playoff_win_pct = {name: 0 for name in df_team_stats.index}
    top4_pct = {name: 0 for name in df_team_stats.index}
    results_list = []
    for _ in range(run_times):
        proj_win_df = df_team_stats[['wins', 'points_for', 'points_against']]

        for j in range(len(df_proj)):
            margin = df_proj.loc[j, 'average_projections'] + np.random.normal(0, 12)
            if margin > 0:
                proj_win_df.loc[df_proj.loc[j, 'Home Team'], 'wins'] += 1

            else:
                proj_win_df.loc[df_proj.loc[j, 'Away Team'], 'wins'] += 1

            proj_win_df.loc[df_proj.loc[j, 'Home Team'], 'points_for'] += margin
            proj_win_df.loc[df_proj.loc[j, 'Away Team'], 'points_for'] -= margin

        proj_win_df['pts_diff'] = proj_win_df['points_for'] - proj_win_df['points_against']
        proj_win_df = proj_win_df.sort_values(['wins', 'pts_diff'], ascending=False)

        for r in proj_win_df[:4].index:
            top4_pct[r] += 1
            playoff_win_pct[r] += 1

        for r in proj_win_df[4:8].index:
            playoff_win_pct[r] += 1

        season_dict = {name:[proj_win_df.loc[name,'wins'],proj_win_df.loc[name,'pts_diff']] for name in proj_win_df.index}
        results_list.append(season_dict)

    return playoff_win_pct, top4_pct, results_list


def view_weekly_projections(df_scores, df_team_stats):

    df_proj = df_scores.copy()
    df_proj = df_proj[df_proj['Home Score'].isnull()]
    df_proj['Date'] = pd.to_datetime(df_proj['Date'], format='%Y/%m/%d')

    df_proj = df_proj[df_proj['Date'] <= (datetime.datetime.now() + datetime.timedelta(days=7))]
    df_proj = df_proj.drop(['Home Score', 'Away Score'], axis=1)
    df_proj = df_proj.reset_index(drop=True)

    df_proj = set_projections(df_proj, df_team_stats)

    for i in range(df_proj.shape[0]):
        print('Home Team: ' + df_proj.loc[i, 'Home Team'], end='')
        print(' vs Away Team: ' + df_proj.loc[i, 'Away Team'])

        print('Elo Margin: \t' + str(round(df_proj.loc[i, 'elo_margin'],2)), end='')
        print('\tHome Chance: ' + str(round(df_proj.loc[i, 'elo_home_chance'],2)), end='')
        print('\tAway Chance: ' + str(round(df_proj.loc[i, 'elo_away_chance'],2)))

        print('DVOA Margin: \t' + str(round(df_proj.loc[i, 'dvoa_margin'],2)), end='')
        print('\tHome Chance: ' + str(round(df_proj.loc[i, 'home_dvoa_win%'],2)), end='')
        print('\tAway Chance: ' + str(round(df_proj.loc[i, 'away_dvoa_win%'],2)))

        print('DVOHAA Margin: \t' + str(round(df_proj.loc[i, 'dvhaa_margin'],2)), end='')
        print('\tHome Chance: ' + str(round(df_proj.loc[i, 'home_dvhaa_win%'],2)), end='')
        print('\tAway Chance: ' + str(round(df_proj.loc[i, 'away_dvhaa_win%'],2)))

        print('Avg Margin: \t' + str(round(df_proj.loc[i, 'average_projections'],2)), end='')
        print('\tHome Chance: ' + str(round(df_proj.loc[i, 'home_avg_win%'],2)), end='')
        print('\tAway Chance: ' + str(round(df_proj.loc[i, 'away_avg_win%'],2)))
        print()

    return 0


def view_average_season_projections(df_team_stats, average_projections):

    df_team_stats['Projected_wins'] = df_team_stats.index.map(average_projections)
    df_team_stats['Projected_wins'] += df_team_stats['wins']

    for team in df_team_stats.index:
        print(team, end='\n\t\t')
        print(round(df_team_stats.loc[team, 'Projected_wins'], 2))

    return 0


def view_future_projections(df_scores, df_team_stats):

    df_proj = df_scores.copy()
    df_proj = df_proj[df_proj['Home Score'].isnull()]
    df_proj['Date'] = pd.to_datetime(df_proj['Date'], format='%Y/%m/%d')

    df_proj = df_proj.drop(['Home Score', 'Away Score'], axis=1)
    df_proj = df_proj.reset_index(drop=True)

    df_proj = set_projections(df_proj, df_team_stats)

    option_str = 'Future Projections\n(1) View Average Results\n'
    option_str += '(2) Run Monte Carlo Simulations\n'

    input_code = input(option_str)

    if input_code == '1':

        average_projections = dict(df_proj.groupby(['Home Team'])['home_avg_win%'].sum() +
                              df_proj.groupby(['Away Team'])['away_avg_win%'].sum())

        view_average_season_projections(df_team_stats, average_projections)

    elif input_code == '2':

        input_run = input('How many simulations? ')

        run_times = int(input_run)

        finals_pct, top4_pct, results_list = simulate_season(df_proj, df_team_stats, run_times)
        df_season_mc = pd.DataFrame.from_records(results_list)
        df_season_mc.to_csv('nrl_2023_mc.csv')

        for t in finals_pct:
            print(f'{t}: \n\tFinals: {(finals_pct[t] / run_times):.2%}\t\tTop 4: {(top4_pct[t] / run_times):.2%}')

    return 0


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    # Read 2023 NRL data and cleans to ensure correct team names
    df_nrl_2023 = clean_nrl_scores('2023_nrl.csv')
    updated_models = False

    option_str = 'NRL Model and Projections\n(1) Enter New Scores\n'
    option_str += '(2) View Weekly Projections\n'
    option_str += '(3) View Future Projections\n(exit) Exit\n\nEnter Option: '

    input_code = input(option_str)

    while input_code != 'exit':

        # Enter New Scores
        if input_code == '1':

            df_nrl_2023 = enter_new_scores(df_nrl_2023)
            df_nrl_2023.to_csv('2023_nrl.csv', index=False)

        elif input_code == '2':

            df_team_stats = update_models(df_nrl_2023)
            view_weekly_projections(df_nrl_2023, df_team_stats)

        elif input_code == '3':

            df_team_stats = update_models(df_nrl_2023)
            view_future_projections(df_nrl_2023, df_team_stats)

        else:

            print('Not a valid option\n{1,2,3 or \'exit\'')
        # Update Models and Projections
        # Weekly Projections
        # Future Projections
        input_code = input('\n'+option_str)
