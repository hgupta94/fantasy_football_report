import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import requests
import os
import datetime as dt
import random
from functools import reduce
import math
import itertools
import json
import sys
from scipy.stats import iqr

# %% Data Setup
def load_data(league_id, season, swid="", espn=""):
    '''
    Pull ESPN API data for a particular season (current API version only goes back to 2018).
    '''

    url = 'http://fantasy.espn.com/apis/v3/games/ffl/seasons/' + str(int(season)) + '/segments/0/leagues/' + str(int(league_id)) + '?view=mMatchupScore&view=mTeam&view=mSettings'
    r = requests.get(url,
                     cookies = {'SWID': swid,
                                'espn_s2': espn},
                                params={'view':'mMatchup'})
    d = r.json()

    #json = json.dumps(d)
    return d

def get_params(d):
    '''
    Returns general league information used throughout the analysis
    '''
    params = {}

    # general league setup
    league_size = d['settings']['size']
    roster_size = sum(d['settings']['rosterSettings']['lineupSlotCounts'].values())

    regular_season_end = d['settings']['scheduleSettings']['matchupPeriodCount']
    current_week = d['scoringPeriodId'] - 1
    matchup_week = d['scoringPeriodId']
    weeks_left = regular_season_end - current_week
    playoff_teams = d['settings']['scheduleSettings']['playoffTeamCount']
    playoff_matchup_length = d['settings']['scheduleSettings']['playoffMatchupPeriodLength']

    # roster construction
    # need to figure out rest of position codes
    slotcodes = {
        0: 'QB', 1: 'QB',
        2: 'RB', 3: 'Flex',
        4: 'WR', 5: 'Flex',
        6: 'TE',
        10: 'LB', 11: 'DL',
        14: 'DB',
        16: 'DST', 17: 'K',
        18: 'P', 19: 'HC',
        20: 'Bench', 21: 'IR',
        23: 'Flex'}
    lineup_slots = d['settings']['rosterSettings']['lineupSlotCounts']
    lineup_slots_df = pd.DataFrame.from_dict(lineup_slots, orient='index').rename(columns={0:'limit'})
    lineup_slots_df['posID'] = lineup_slots_df.index.astype('int')
    lineup_slots_df = lineup_slots_df[lineup_slots_df.limit > 0]
    lineup_slots_df['pos'] = lineup_slots_df.replace({'posID': slotcodes}).posID

    # Mapping team ID to team name
    prim_owner = [[game['primaryOwner'], game['id']] for game in d['teams']]
    prim_owner = pd.DataFrame(prim_owner).rename({0:"id_string", 1:"id"}, axis=1)
    owner_map = [[game['id'], game['firstName'], game['lastName'][:1]] for game in d['members']]
    owner_map = pd.DataFrame(owner_map).rename({0:"id_string", 1:"first_nm", 2:"last_init"}, axis=1)
    owner_map["team"] = owner_map.first_nm + " " + owner_map.last_init
    team2 = pd.merge(prim_owner, owner_map, on="id_string")
    team2["team"] = team2.team.str.title().str.replace(" ", "")
    team_map2 = dict(zip(team2.id, team2.team))

    # Get weekly matchups
    df = pd.DataFrame()
    for game in d['schedule']:
        if game['matchupPeriodId'] > d['settings']['scheduleSettings']['matchupPeriodCount']:
            continue
        if game['matchupPeriodId'] <= d['settings']['scheduleSettings']['matchupPeriodCount']:
            week = game['matchupPeriodId']
            team1 = game['home']['teamId']
            score1 = game['home']['totalPoints']
            team2 = game['away']['teamId']
            score2 = game['away']['totalPoints']
            matchups = pd.DataFrame([[week, team1, score1, team2, score2]],
                                    columns=["week", "team1", "score1", "team2", "score2"])
            df = df.append(matchups)
    df = df.replace({'team1':team_map2, 'team2':team_map2})

    #position = lineup_slots_df.pos.str.lower().drop(labels=['20','21']).tolist()
    position = lineup_slots_df.pos.str.lower().to_list()
    position = np.setdiff1d(position, ['bench', 'ir']).tolist()
    teams = list(team_map2.values())

    # get FAAB budget remaining
    faab = d["settings"]["acquisitionSettings"]["acquisitionBudget"]
    budget = pd.DataFrame()
    for tm in d["teams"]:
        teamid = tm["id"]
        faab_left = faab - tm["transactionCounter"]["acquisitionBudgetSpent"]
        df_tm = pd.DataFrame([[teamid, faab_left]],
                          columns=["team", "faab_left"])
        budget = budget.append(df_tm)
    budget = budget.replace({"team":team_map2})


    params = {"league_size": league_size,
              "roster_size": roster_size,
              "regular_season_end": regular_season_end,
              "current_week": current_week,
              "matchup_week": matchup_week,
              "weeks_left": weeks_left,
              "playoff_teams": playoff_teams,
              "team_map": team_map2,
              "matchup_df": df,
              "positions": position,
              "teams": teams,
              "slotcodes": slotcodes,
              "lineup_slots_df": lineup_slots_df,
              "playoff_matchup_length": playoff_matchup_length,
              "faab_remaining": budget
              }

    return params

def power_rank(d, league_id, season, week=None):
    '''
    Calculates a weekly power ranking and power score for each team
    '''

    # return parameters
    regular_season_end = get_params(d)["regular_season_end"]
    current_week = get_params(d)["current_week"]
    matchup_week = get_params(d)["matchup_week"]

    # get weekly matchups and scores
    df = get_params(d)["matchup_df"]

    # Calculate W/L
    df['team1_result'] = np.where(df['score1'] > df['score2'], 1, 0)
    df['team2_result'] = np.where(df['score2'] > df['score1'], 1, 0)

    # Account for ties
    mask = (df.score1 == df.score2) & (df.score1 > 0) & (df.score2 > 0)
    df.loc[mask, ['team1_result', 'team2_result']] = 0.5

    # convert dataframe to long format so each row is a team week, not matchup
    home = df.iloc[:,[0,1,2,5]].rename(columns={'team1':'team', 'score1':'score', 'team1_result':'result'})
    home['id'] = home['team'].astype(str) + home['week'].astype(str)
    away = df.iloc[:,[0,3,4,6]].rename(columns={'team2':'team', 'score2':'score', 'team2_result':'result'})
    away['id'] = away['team'].astype(str) + away['week'].astype(str)

    df_current = pd.concat([home, away])
    season_wins = (df_current.groupby(['team', 'week'])
                        .sum()
                        .groupby(level=0)
                        .cumsum()
                        .rename(columns={'result':'wins'})
                        .reset_index())
    season_wins['id'] = season_wins['team'].astype(str) + season_wins['week'].astype(str)
    season_wins = season_wins.drop(['team', 'week', 'score'], axis=1)
    df_current = pd.merge(df_current, season_wins, on='id')


    # Calculate total season points by team
    cumul_score = (df_current.groupby(['team', 'week'])
                             .sum()
                             .groupby(level=0)
                             .cumsum()
                             .reset_index())
    cumul_score['id'] = cumul_score['team'].astype(str) + cumul_score['week'].astype(str)

    cumul_score = (cumul_score.drop(['team', 'week', 'wins', 'result'], axis=1)
                             .rename(columns={'score':'total_pf'}))

    df_current = pd.merge(df_current, cumul_score, on='id')
    df_current['ppg'] = df_current['total_pf'] / df_current['week']


    # Calculate median scores
    median_score = (df_current.groupby(['week'])
                            .agg(np.median)
                            .rename(columns={'score':'score_med', 'total_pf':'total_pf_med', 'ppg':'ppg_med'})
                            .drop(['wins', 'result'], axis=1))
    df_current = pd.merge(df_current, median_score, how='left', on='week')


    # Calculate rolling standard deviation by team
    st_dev = df_current[['week', 'team', 'score', 'result', 'id']]
    st_dev = (st_dev.set_index('week')
            .sort_index()
            .groupby(['team'])['score']
            .expanding()
            .std()
            .to_frame()
            .rename(columns={'score':'sd'})
            .reset_index())
    st_dev['id'] = st_dev['team'].astype(str) + st_dev['week'].astype(str)
    df_current = st_dev.drop(['team','week'], axis=1).fillna(0).merge(df_current, on='id')

    # Calculate luck factor
    # actual w/l result vs your week score rank
    # positive means more lucky (win but lower score)
    # negative means more UNlucky (loss but higher score)
    df_current['week_luck'] = df_current['result'] - (df_current.groupby('week')['score'].rank() - 1) / 9
    luck_sum = (df_current.groupby(['team', 'week'])
                        .sum()
                        .groupby(level=0)
                        .cumsum()
                        .reset_index()
                        .loc[:,['team','week','week_luck']])
    luck_sum['luck_index'] = luck_sum['week_luck'] / luck_sum['week']
    luck_sum['id'] = luck_sum['team'].astype(str) + luck_sum['week'].astype(str)
    df_current = luck_sum.drop(['team', 'week', 'week_luck'], axis=1).merge(df_current, on='id')

    # Calculate ranking metrics
    exp_win = df_current[['week', 'team', 'result', 'week_luck']]
    exp_win['xwins'] = exp_win['result'] - exp_win['week_luck']
    exp_win = (exp_win.groupby(['team', 'week'])
              .sum()
              .groupby(level=0)
              .cumsum()
              .reset_index()
              .drop(['result', 'week_luck'], axis=1))
    exp_win['id'] = exp_win['team'].astype(str) + exp_win['week'].astype(str)
    df_current = exp_win.drop(['team', 'week'], axis=1).merge(df_current, on='id')

    three_wk_avg = (df_current.groupby('team')['score']
                    .transform(lambda x: x.rolling(3, 1)
                    .mean())
                    .rename('three_wk_avg'))
    df_current = pd.merge(df_current, three_wk_avg, left_index = True, right_index=True)

    three_wk_med = (df_current[['week', 'three_wk_avg']]
                    .groupby('week')
                    .agg(np.median)
                    .reset_index()
                    .rename(columns={'three_wk_avg':'three_wk_med'}))
    df_current = pd.merge(df_current, three_wk_med, how='left', on='week')

    # calculate indexes for power score
    df_current['win_index'] = (df_current['wins'] / (df_current['week']))
    df_current['score_index'] = (df_current['three_wk_avg'] / df_current['three_wk_med'])
    df_current['season_index'] = (df_current['total_pf'] / df_current['total_pf_med'])
    df_current['consistency'] = (1 - (df_current['sd'] / df_current['ppg'])) * df_current['season_index']

    cons_med = df_current[['week', 'consistency']]
    cons_med = cons_med.groupby('week').agg(np.median).rename(columns={'consistency':'consistency_med'})
    df_current = pd.merge(df_current, cons_med, on='week')
    df_current['consistency_index'] = df_current['consistency'] / df_current['consistency_med']

    df_current['power_rank_score'] = ((df_current['win_index'] * 0.75)
                                   - (df_current['luck_index'] * 0.2)
                                   + (df_current['consistency_index'] * 1.2)
                                   + (df_current['score_index'] * 0.9)
                                   + (df_current['season_index'] * 1.2))

    # Standardize so average=100
    pr_avg = df_current.loc[:,['week', 'power_rank_score']]
    pr_avg = pr_avg.groupby('week').agg(np.mean).rename(columns={'power_rank_score':'power_rank_avg'})
    df_current = pd.merge(df_current, pr_avg, on='week')
    df_current['power_score'] = (df_current.power_rank_score / df_current.power_rank_avg) * 100
    df_current['power_rank'] = df_current.groupby(['week'])['power_rank_score'].rank(ascending=False)
    df_current['rank_change'] = df_current.groupby(['team'])['power_rank'].diff() * -1
    df_current['rank_change'] = df_current.rank_change.fillna(0)
    df_current['score_change'] = df_current.groupby(['team'])['power_score'].diff()
    df_current['score_change'] = df_current.score_change.fillna(0)

    df_current = df_current[df_current.week < matchup_week]

    return df_current

# %% Simulate Current Week

def sim_week(d, positions, matchup_week, league_size, team_map, matchups, teams, slotcodes, pos_df, n_sim=10):
    '''
    Simulates current week matchups using aggregate projections from FantasyPros
    '''

    # get aggregate projections from FantasyPros
    positions.remove("flex")

    projections = pd.DataFrame()
    # Return current week's projections for all positions
    for pos in positions:
        url = 'https://www.fantasypros.com/nfl/projections/' + pos + '.php?scoring=HALF&week=' + str(matchup_week)
        df = pd.read_html(url)[0]

        # drop multi index column
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel()

        df['POSITION'] = pos
        df = df[['Player', 'FPTS', 'POSITION']]

        # remove team from player name
        if pos != 'dst':
            df['Player'] = df['Player'].str[:-3]
            df['Player'] = df['Player'].str.rstrip()

        if pos == 'dst':
            df['Player'] = df['Player'].str.split().str[-1] + ' DST'

        df['Player'] = df.Player.str.lower()
        df.columns = df.columns.str.lower()
        df['player'] = df['player'].str.replace('.', '', regex=True)
        df['player'] = df['player'].str.replace('\'', '', regex=True)
        df['player'] = df['player'].str.replace(' iii', '', regex=True)
        df['player'] = df['player'].str.replace(' ii', '', regex=True)
        df['player'] = df['player'].str.replace(' jr', '', regex=True)

        projections = projections.append(df)

    # fix wft name
    projections['player'] = projections['player'].replace(['team dst'],'washington dst')

    # load parameters
    df = matchups
    posns = pos_df.pos.str.lower().to_list()
    struc = pos_df.limit.to_list()

    # get actual and projected scores from espn
    # code thanks to Steven Morse: https://stmorse.github.io/journal/espn-fantasy-projections.html
    data = []
    week = matchup_week

    for tm in d['teams']:
        tmid = tm['id']
        for p in tm['roster']['entries']:
            name = p['playerPoolEntry']['player']['fullName']
            slot_id = p['lineupSlotId']
            slot  = slotcodes[slot_id]

            # injured status (try/except bc dst cannot be injured)
            inj = 'NA'
            try:
                inj = p['playerPoolEntry']['player']['injuryStatus']
            except:
                pass

            # projected/actual points
            proj, act = None, None
            for stat in p['playerPoolEntry']['player']['stats']:
                if stat['scoringPeriodId'] != week:
                    continue
                if stat['statSourceId'] == 0:
                    act = stat['appliedTotal']
                elif stat['statSourceId'] == 1:
                    proj = stat['appliedTotal']

            data.append([
                week, tmid, name, slot_id, slot, inj, proj, act
            ])

    proj = pd.DataFrame(data,
                        columns=['week', 'team', 'player', 'slot_id',
                                 'slot', 'status', 'proj', 'actual'])
    proj = proj.apply(lambda x: x.astype(str).str.lower() if x.dtype=='object' else x)
    proj = proj.replace({"team": team_map})
    proj = proj[(proj['slot'] != 'ir')]
    proj = proj.drop(['proj'], axis=1)
    proj['actual'] = np.where(proj['actual'] == 'none', np.nan, proj['actual'])

    # fix player names
    proj['player'] = proj['player'].str.replace('.', '', regex=True)
    proj['player'] = proj['player'].str.replace('\'', '', regex=True)
    proj['player'] = proj['player'].str.replace('/', '', regex=True)
    proj['player'] = proj['player'].str.replace(' iii', '', regex=True)
    proj['player'] = proj['player'].str.replace(' ii', '', regex=True)
    proj['player'] = proj['player'].str.replace(' jr', '', regex=True)

    # replace espn projections with aggregate projections
    proj = pd.merge(proj, projections, how='left', on='player').rename(columns={'fpts':'projected'})
    proj['projected'] = np.where(proj['projected'].isnull(), 0, proj['projected'])

    # add standard deviations (same as get_ros_projections)
    proj['sd'] = np.where(proj['position'] == 'qb', proj['projected']*0.2, proj['projected']*0.4)

    # get current week matchups
    matchups = df[['week', 'team1', 'score1', 'team2', 'score2']]
    matchups = matchups[matchups['week'] == matchup_week]
    matchups['team1'] = matchups.team1
    matchups['team2'] = matchups.team2
    matchups['game_id'] = range(1, int(league_size/2)+1)
    matchups = matchups.reset_index(drop=True)

    # initialize dicionaries for counts
    teams = list(team_map.values())
    teams_dict = {key: 0 for key in teams}
    n_wins = {key: 0 for key in teams}
    n_highest = {key: 0 for key in teams}
    n_lowest = {key: 0 for key in teams}

    # simulate current week scores
    projections["sd"] = np.where(projections['fpts'] == 'qb', projections['fpts']*0.2, projections['fpts']*0.4)
    ref = projections[["position", "fpts", "sd"]].groupby("position").quantile(0.75).reset_index()
    score_df = pd.DataFrame(columns={'team', 'score'})
    for sim in range(n_sim):
        # get starting lineeup of highest projected player by position
        # for each team, select top: 1QB, 2RB, 2WR, 1TE, 1FLEX, 1DST, 1K
        for_sim = pd.DataFrame()
        for tm in teams:
            for pos, num in zip(posns, struc):
                if pos != "flex":
                    starter = proj.query('team == @tm & slot == @pos & ~actual.isnull()')
                    for_sim = for_sim.append(starter)
                    if len(starter) < num:
                        selection = proj.query('team == @tm & position == @pos & actual.isnull()').sort_values(by="projected", ascending=False).head(num-len(starter))
                        selection['proj_slot'] = pos
                        if len(selection) < num - len(starter):
                            # use 75th percentile player if position is not filled
                            num_left = num - len(selection)
                            ref_proj = ref.loc[ref.query('position == @pos').index.repeat(num_left)].rename(columns={"fpts":"projected"})
                            ref_proj["team"] = tm
                            ref_proj['proj_slot'] = pos
                            for_sim = for_sim.append(ref_proj)
                        else:
                            for_sim = for_sim.append(selection)
                    pl_list = for_sim.player.tolist()

                if pos == "flex":
                    fl_starter = proj.query('team == @tm & slot == @pos & ~actual.isnull()')
                    if len(fl_starter) < num:
                        # select flex: 3rd RB/WR or 2nd TE
                        fpos = ["rb", "wr"]
                        fnum = [2, 2]
                        flex = pd.DataFrame()
                        for a, b in zip(fpos, fnum):
                            selection = proj.query('team == @ tm & position == @a & actual.isnull()').sort_values(by="projected", ascending=False)
                            selection = selection[~selection.player.isin(pl_list)]
                            selection = selection.groupby("position").head(1)
                            flex = flex.append(selection)
                        # select flex player
                        flex = flex.sort_values(by="projected", ascending=False).head(1)
                        flex['proj_slot'] = pos
                        for_sim = for_sim.append(flex)
                    else:
                        for_sim = for_sim.append(fl_starter)
        for_sim = for_sim.reset_index(drop=True)

        # if player hasn't played, simulate score otherwise use actual scores
        for index, row in for_sim['actual'].iteritems():
            try:
                if pd.isnull(row):
                    for_sim.at[index,'score'] = ((random.gauss(for_sim['projected'][index], for_sim['sd'][index]))
                                               + (random.uniform(-1, 1))) * 0.97
                else:
                    for_sim.at[index,'score'] = for_sim['actual'][index]
            except:
                pass

        for team in teams:
            teams_dict[team] = for_sim[for_sim['team'] == team].score.sum()
            score_df = score_df.append(for_sim[for_sim['team'] == team].groupby('team').score.sum().reset_index())

        a = matchups.filter(like='team').columns
        matchups['score' + a.str.lstrip('team')] = matchups[a].stack().map(teams_dict).unstack()

        # calculate wins and losses
        matchups['team1_result'] = np.where(matchups['score1'] > matchups['score2'], 1, 0)
        matchups['team2_result'] = np.where(matchups['score2'] > matchups['score1'], 1, 0)

        # account for ties
        mask = (matchups.score1 == matchups.score2)
        matchups.loc[mask, ['team1_result', 'team2_result']] = 0.5

        # convert dataframe to long format so each row is a team week, not matchup
        home = matchups.iloc[:,[0,1,2,5,6]].rename(columns={'team1':'team', 'score1':'score', 'team1_result':'wins'})
        away = matchups.iloc[:,[0,3,4,5,7]].rename(columns={'team2':'team', 'score2':'score', 'team2_result':'wins'})
        df_sim = pd.concat([home, away]).iloc[:,[1,2,3,4]]

        for team in teams:
            n_wins[team] += df_sim[df_sim['team'] == team].wins.values[0].astype(int)

        # get highest/lowest scorer and teams in top half (need to add this)
        high = df_sim.sort_values(by='score', ascending=False).iloc[0,0]
        low = df_sim.sort_values(by='score').iloc[0,0]
        n_highest[high] += 1
        n_lowest[low] += 1

    # convert dicts to df and combine
    game_id = df_sim.loc[:,['team', 'game_id']]
    wins = pd.DataFrame(n_wins.items(), columns=['team', 'n_wins'])
    highest = pd.DataFrame(n_highest.items(), columns=['team', 'n_highest'])
    lowest = pd.DataFrame(n_lowest.items(), columns=['team', 'n_lowest'])

    dfs = [game_id, wins, highest, lowest]

    week_sim = reduce(lambda left, right: pd.merge(left, right, on='team'), dfs)

    return week_sim, score_df

#%% Simulate Season

def get_ros_proj(d, current_week, matchup_week, league_size, roster_size, team_map):
    '''
    Returns rest of season projections for all rostered players
    Simulates a weekly score for each team using current roster.
    Used for season and playoff simulations, NOT current week.
    This *should* be flexible enough to fit almost every league.
    '''

    # Map players to teams
    players = []
    teams = []

    for i in range(league_size):
        for player in range(roster_size):
            try:
                players.append(d['teams'][i]['roster']['entries'][player]['playerPoolEntry']['player']['fullName'])
                teams.append(d['teams'][i]['id'])
            except IndexError:
                pass

    # Fix player names
    rosters = pd.DataFrame({'team': teams, 'player': players}).apply(lambda x: x.astype(str).str.lower())
    rosters = rosters.replace({'player':[',', '\(', '\)', '\'', ' jr.', '\.']}, {'player':''}, regex=True)
    rosters = rosters.replace(['michael '], 'mike ', regex=True)

    rosters = rosters.replace('william fuller v', 'will fuller')
    rosters['player'] = rosters['player'].str.replace(' iii', '')
    rosters['player'] = rosters['player'].str.replace(' ii', '')
    rosters['player'] = rosters['player'].str.replace(' iv', '')
    #rosters['player'] = rosters['player'].str.replace(' v', '')
    rosters['player'] = rosters['player'].str.replace('d/st', 'dst')

    # Add rest of season projections to players
    positions = get_params(d)["positions"]
    positions.remove("flex")
    positions = ['d' if i=='dst' else i for i in positions]
    d_map = {"Los Angeles Rams":"Rams", "Tampa Bay":"Buccaneers", "Buffalo":"Bills", "Miami":"Dolphins",
             "San Francisco":"49ers", "Dallas":"Cowboys", "Jacksonville":"Jaguars", "Washington":"Washington",
             "New England":"Patriots", "Baltimore":"Ravens", "Green Bay":"Packers", "Philadelphia":"Eagles",
             "New Orleans":"Saints", "Kansas City":"Chiefs", "New York Giants":"Giants", "New York Jets":"Jets",
             "Arizona":"Cardinals", "Pittsburgh":"Steelers", "Seattle":"Seahawks", "Carolina":"Panthers",
             "Atlanta":"Falcons", "Cleveland":"Browns", "Indianapolis":"Colts", "Denver":"Broncos",
             "Los Angeles Chargers":"Chargers", "Tennessee":"Titans", "Chicago":"Bears", "Minnesota":"Vikings",
             "Cincinnati":"Bengals", "Houston":"Texans", "Las Vegas":"Raiders", "Detroit":"Lions"}

    player_data = pd.DataFrame()
    proj = pd.DataFrame()
    for pos in positions:
        url = 'https://www.numberfire.com/nfl/fantasy/remaining-projections' + '/' + pos

        data = pd.read_html(url, header=None)[0]
        data = pd.DataFrame(np.vstack([data.columns, data]))
        data.columns = ["player"]
        if pos == "d":
            # replace DST with mascot name
            dst = data.player.str.split(" D/ST", expand=True)
            dst = dst.replace({0:d_map})
            dst[1] = "DST"
            dst["player"] = dst[0] + " " + dst[1]
            data = dst.player.to_frame()
        player_data = player_data.append(data, ignore_index=True)

        df = pd.read_html(url)[1]
        proj = proj.append(df, ignore_index=True)

    # Fix player names
    player_data[player_data.player.str.startswith("Amon-Ra")] = "Amon-Ra StBrown  A. St.Brown  WR DET"
    player_data[player_data.player.str.startswith("Equanimeous")] = "Equanimeous StBrown  E. St.Brown  WR GB"
    player_data = player_data.replace([',', '\(', '\)', '\'', ' Jr.', '\.', ' III', ' II', ' IV'], '', regex=True)
    player_data = player_data.replace(['Michael '],'Mike ', regex=True)
    player_data = player_data.iloc[:,0].str.split(expand=True)
    player_data['name'] = player_data[0].str.cat(player_data[1],sep=" ")

    proj.columns = proj.columns.get_level_values(1)
    proj = proj.fillna(0)
    proj["FP"] = proj.FP + (proj.Rec * 0.5)  # add half ppr scoring
    proj = proj.loc[:,"FP"].to_frame()

    # Combine datasets
    player_projections = pd.concat([player_data.iloc[:,[6,4,5]], proj], axis=1)
    player_projections.columns = ['player', 'position', "team", 'ros_proj']
    player_projections["position"].fillna("DST", inplace=True)
    player_projections = player_projections.apply(lambda x: x.astype(str).str.lower())

    # fix ESB and ARSB
    player_projections["player"][player_projections.player.str.startswith("amon-ra")] = "amon-ra st brown"
    player_projections["player"][player_projections.player.str.startswith("equanimeous")] = "equanimeous st brown"

    # get team values for defenses to merge on later
    player_projections['team'] = (np.where(player_projections.team == "none",
                      player_projections['player'].apply(lambda x: x.split(" ")[0]),
                      player_projections['team']))

    # fix team names to match CSV
    player_projections["team"] = player_projections.team.replace({"jac": "jax", "la":"lar", "wsh":"was"})

    # get injury data and expected return
    injuries = pd.DataFrame()
    df = pd.read_html("https://www.vegasinsider.com/nfl/injuries/")
    for i in range(len(df)):
        if len(df[i].columns) == 6:
            df[i] = df[i].rename(columns=df[i].iloc[0]).iloc[1]
            injuries = injuries.append(df[i])
    injuries = injuries[injuries.Pos.str.lower().isin(positions)]
    injuries["Player"] = injuries.Player.replace([',', '\(', '\)', '\'', ' Jr.', '\.', ' III', ' II', ' IV'], '', regex=True).str.lower()
    injuries["Pos"] = injuries.Pos.str.lower()
    injuries["Expected Return"] = pd.to_datetime(injuries["Expected Return"])
    injuries["wks_out"] = (injuries["Expected Return"] - pd.to_datetime(dt.datetime.now())).dt.days / 7
    injuries["wks_out"] = np.floor(injuries.wks_out)
    injuries["return_wk"] = injuries.wks_out + matchup_week
    injuries = injuries[["Player", "Pos", "wks_out", "return_wk"]]

    # get team  bye weeks
    tm_map = pd.read_csv("https://gist.githubusercontent.com/cnizzardini/13d0a072adb35a0d5817/raw/dbda01dcd8c86101e68cbc9fbe05e0aa6ca0305b/nfl_teams.csv").iloc[:,[1,2]]
    tm_map["Name"] = tm_map.Name.replace({'NY':'New York'}, regex=True)
    byes = pd.read_html("https://www.footballdiehards.com/nfl-bye-weeks.cfm")[0].dropna()
    byes = byes.replace({"Giants Giants":"Giants", "Jets Jets":"Jets"}, regex=True)
    byes = byes.assign(teams=byes['Teams'].str.split(',')).explode('teams').drop("Teams", axis=1)
    byes["teams"] = byes.teams.str.strip()   # remove whitespace
    byes = pd.merge(byes, tm_map, left_on="teams", right_on="Name")
    byes['dst_name'] = byes['Name'].apply(lambda x: x.split(" ")[-1])
    byes.loc[byes.dst_name == "Team", "dst_name"] = "Washington"    # replace WFT name
    byes["week"] = byes['week'].apply(lambda x: x.split(" ")[-1]).astype(int)
    byes["Abbreviation"] = byes.Abbreviation.str.lower()
    byes["dst_name"] = byes.dst_name.str.lower()
    byes = byes.rename({"week":"bye"}, axis=1)
    byes = byes[["bye", "teams", "Abbreviation", "dst_name"]]

    # merge byes with player projections
    pl_byes = pd.merge(player_projections, byes, left_on="team", right_on="Abbreviation")
    # merge dst byes
    dst_byes = pd.merge(player_projections[player_projections.position == "dst"], byes, left_on="team", right_on="dst_name")
    player_projections = pl_byes.append(dst_byes, ignore_index=True)

    # merge injured players
    player_projections = pd.merge(player_projections, injuries, left_on=["player", "position"], right_on=["Player", "Pos"], how="left")
    player_projections = player_projections[["player", "position", "ros_proj", "bye", "wks_out", "return_wk"]]
    player_projections['ros_proj'] = player_projections.ros_proj.astype(float)

    # Calculate points per game
    player_projections["bye_occ"] = np.where(player_projections.bye >= current_week, 1, 0)    # is the bye week in the future?
    player_projections["wks_out"] = player_projections.wks_out.fillna(0)
    player_projections["gms_rem"] = 18 - current_week - player_projections.bye_occ - player_projections.wks_out
    player_projections['ppg'] = player_projections.ros_proj / player_projections.gms_rem

    # set standard deviations
    # qb's have less variance than other positions
    player_projections['sd'] = np.where(player_projections['position'] == 'qb', player_projections['ppg']*0.2, player_projections['ppg']*0.4)

    team_projections = pd.merge(rosters, player_projections, how='left', on='player')
    team_projections["team"] = team_projections.team.astype(int)
    team_projections = team_projections.replace({"team":team_map})

    params = [team_projections, player_projections]

    return params

def sim_scores(d, df_current, current_week, curr_sim_wk, teams, pos_df, ros):
    # get parameters
    posns = pos_df.pos.str.lower().to_list()
    struc = pos_df.limit.to_list()
    ref = ros[1][["position", "ppg", "sd"]].groupby("position").quantile(0.75).reset_index()

    # if current sim week is a bye or player is injured, set proj ppg to 0
    team_ros = ros[0]
    team_ros["ppg"] = team_ros.ros_proj / team_ros.gms_rem
    team_ros["ppg"] = np.where((curr_sim_wk == team_ros.bye) | (curr_sim_wk < team_ros.return_wk),
                                        0,
                                        team_ros.ppg)
    team_ros = team_ros[team_ros.ppg >= 4]


    # assign weights for projections
    if current_week < 4:
        ros_wt = 0.65
        rand_wt = 0.35
    else:
        season_wt = 0.15
        ros_wt = 0.5
        rand_wt = 0.35

    # start simulation
    score_proj_df = pd.DataFrame()
    for tm in teams:
        pts = {'qb.proj':0, 'rb.proj':0,
               'wr.proj':0, 'te.proj':0,
               'dst.proj':0, 'dl.proj':0,
               'lb.proj':0, 'db.proj':0,
               'hc.proj':0, 'p.proj':0,
               'flex.proj':0, 'k.proj':0}

        # simulate using actual season data
        team_stats = df_current[(df_current['week'] == (current_week)) & (df_current['team'] == tm)].reset_index(drop=True)
        team_stats = team_stats[['team', 'ppg', 'sd']]

        # simulate using mean and sd, then add weight
        if current_week < 4:
            season_proj = 0
        else:
            season_proj = random.gauss(team_stats['ppg'], team_stats['sd'])[0] * season_wt

        # simulate through each position
        for pos, num in zip(posns, struc):
            if pos == "flex":
                # get 3rd highest rb/wr and 2nd highest te
                fpos = ["rb", "wr"]
                fnum = [2, 2]
                flex = pd.DataFrame()
                for a, b in zip(fpos, fnum):
                    df = team_ros.query('team == @ tm & position == @a').sort_values(by="ppg", ascending=False)
                    df = df.groupby("position").nth(b)
                    if df.empty:
                        # add "free agent" reference player if flex position is empty
                        ref_flex = ref.query('position == @a')
                        flex = flex.append(ref_flex)
                    else:
                        flex = flex.append(df)
                # select random flex player
                flex_play = flex.sample(n=1)
                flex_proj = flex_play.ppg
                flex_sd = flex_play.sd
                pts[pos+'.proj'] = np.random.normal(flex_proj, flex_sd)[0]
            else:
                week_proj = team_ros.query('team == @tm & position == @pos').sort_values(by=["ppg"], ascending=False)
                if len(week_proj) < num:
                    num_left = num - len(week_proj)
                    # take 75th percentile player if position is not filled
                    ref_proj = ref.loc[ref.query('position == @pos').index.repeat(num_left)]
                    week_proj = week_proj.append(ref_proj)
                    proj = week_proj.ppg
                    sd = week_proj.sd
                    pts[pos+'.proj'] = np.random.normal(proj, sd).sum()
                else:
                    proj = week_proj[:num].ppg
                    sd = team_ros.query('team == @tm & position == @pos').sort_values(by=["ppg"], ascending=False).sd[:num]
                    pts[pos+'.proj'] = np.random.normal(proj, sd).sum()

        # weight scores
        ros_proj = sum(pts.values()) * ros_wt

        # add general randomness, because fantasy football be like that
        rand_pts = random.uniform(90, 130) * rand_wt

        # combine projections
        total_proj_pts = season_proj + ros_proj + rand_pts
        row = {'team':tm,'score': total_proj_pts}
        rowdf = pd.DataFrame(data=[row])

        score_proj_df = pd.concat([score_proj_df, rowdf])

    score_proj_df = score_proj_df.reset_index(drop=True)

    return score_proj_df

def sim_matchups(d, df_current, current_week, teams, pos_df, regular_season_end, matchup_week, matchups, ros):
    '''
    Simulates rest of season matchups (head to head) using scores from simulate_scores
    and returns final standings
    '''

    df = matchups

    if matchup_week <= regular_season_end:
        # if current week is not in playoffs, simulate through matchups

        # get weekly matchups
        matchups = df[['week', 'team1', 'score1', 'team2', 'score2']]
        #matchups['team1'] = matchups.team1
        #matchups['team2'] = matchups.team2

        # create separate df for past weeks and append future weeks later
        matchups2 = matchups[matchups['week'] < matchup_week]

        # simulate scores for future weeks
        for curr_sim_wk in range(matchup_week, regular_season_end+1):
            score_sim = sim_scores(d, df_current, current_week, curr_sim_wk, teams, pos_df, ros)
            score_dict = dict(zip(score_sim.team, score_sim.score))
            matchups_new = matchups[matchups['week'] == curr_sim_wk]
            a = matchups_new.filter(like='team').columns
            matchups_new['score' + a.str.lstrip('team')] = matchups_new[a].stack().map(score_dict).unstack()
            matchups2 = matchups2.append(matchups_new)

        # calculate W/L
        matchups2['team1_result'] = np.where(matchups2['score1'] > matchups2['score2'], 1, 0)
        matchups2['team2_result'] = np.where(matchups2['score2'] > matchups2['score1'], 1, 0)

        # account for ties
        mask = (matchups2.score1 == matchups2.score2)
        matchups2.loc[mask, ['team1_result', 'team2_result']] = 0.5

        # convert dataframe to long format so each row is a team week, not matchup
        home = matchups2.iloc[:,[0,1,2,5]].rename(columns={'team1':'team', 'score1':'score', 'team1_result':'wins'})
        away = matchups2.iloc[:,[0,3,4,6]].rename(columns={'team2':'team', 'score2':'score', 'team2_result':'wins'})
        df_sim = pd.concat([home, away]).iloc[:,[1,2,3]]

        final_results = df_sim.groupby('team').agg({'wins':'sum', 'score':'sum'}).reset_index()

        return final_results

    else:
        # if current week is after regular season, return final standings
        data = []
        for tm in d['teams']:
            tmid = tmid = tm['abbrev'].lower()
            wins = tm['record']['overall']['wins']
            losses = tm['record']['overall']['losses']
            record = str(wins) + '-' + str(losses)
            score = tm['record']['overall']['pointsFor']
            pa = tm['record']['overall']['pointsAgainst']
            data.append([tmid, wins, losses, record, score, pa])

        standings = pd.DataFrame(data,
                                 columns = ["team", "wins", "losses", "record", "pf", "pa"])
        standings = standings.sort_values(by=['wins', 'pf'], ascending=False)
        standings = standings[['team', 'record', 'pf', 'pa']]
        standings = standings.rename(columns={"team":"Team", "record":"Record", "pf":"PF", "pa":"PA"})

        return standings

def sim_season(d, league_id, season, current_week, matchup_week, regular_season_end, df_current, teams, matchups, ros, n_sim=1):
    '''
    Simulates regular season and playoffs. Returns number projected standings, win/rank distributions, and playoff projections
    '''

    lineup_slots_df = get_params(d)["lineup_slots_df"]
    pos_df = lineup_slots_df[(lineup_slots_df.posID != 20) & (lineup_slots_df.posID != 21)]

    if matchup_week < regular_season_end:
        # if current week is not in playoffs, simulate season
        # initialize empty table to count wins and ranks
        table = (pd.DataFrame(index=teams,
                               columns=range(regular_season_end + 1))
                .fillna(0)
                .rename(columns={'index':'team'}))

        rank_table = (pd.DataFrame(index=teams,
                               columns=range(1, len(teams) + 1))
                .fillna(0)
                .rename(columns={'index':'team'}))

        # initialize dictionaries to count number of occurances for each team
        n_playoffs = {key: 0 for key in teams}
        n_finals = {key: 0 for key in teams}
        n_champ = {key: 0 for key in teams}
        n_second = {key: 0 for key in teams}
        n_third = {key: 0 for key in teams}

        df = pd.DataFrame()
        for sim in range(n_sim):
            # simulate regular season
            results = sim_matchups(d, df_current, current_week, teams, pos_df, regular_season_end, matchup_week, matchups, ros)
            results = results.set_index('team')

            # top 5 by record, 6th by most points, rest by record
            top5 = results.sort_values(['wins', 'score'], ascending=False).head(5)
            sixth = results[~results.isin(top5)].dropna().sort_values(['score'], ascending=False).head(1)
            playoffs = pd.concat([top5, sixth])
            bot4 = results[~results.isin(playoffs)].dropna().sort_values(['wins'], ascending=False)
            results = pd.concat([playoffs, bot4])
            results['rank'] =  np.arange(len(results)) + 1

            df = df.append(results)
            for index, row in results.iterrows():
                wins = row['wins']
                table[wins][index] += 1

                rank = row['rank']
                rank_table[rank][index] += 1

            # simulate plaoffs
            # get playoff teams
            p_teams = (results.reset_index().sort_values(["rank"])
                       .head(6)
                       .loc[:,"team"])

            # count playoff appearances
            for team in p_teams:
                n_playoffs[team] += 1

            # simulate 1 week of semifinals
            # top 2 teams get bye
            byes = p_teams.head(2).values
            quarter_teams = p_teams.iloc[2:]
            quarter_scores = (sim_scores(d, df_current, current_week, 14, quarter_teams, pos_df, ros)
                              .set_index('team')
                              .reset_index()
                              .rename(columns={0:'score'}))

            # get quarterfinals matchups and winners
            # lower seed needs to win by 10 or more points
            diff = 10
            quarter_1 = quarter_scores.iloc[[0,3],:]
            quarter_1 = np.where(quarter_1.score.iloc[1] - quarter_1.score.iloc[0] > diff, quarter_1.team.iloc[1], quarter_1.team.iloc[0]).astype(object)
            quarter_2 = quarter_scores.iloc[[1,2],:]
            quarter_2 = np.where(quarter_2.score.iloc[1] - quarter_2.score.iloc[0] > diff, quarter_2.team.iloc[1], quarter_2.team.iloc[0]).astype(object)

            # get semifinals matchups and winners
            semi_teams = np.stack((quarter_1.item(), quarter_2.item())).astype(object)
            semi_teams = np.concatenate((byes, semi_teams)).tolist()

            semi_scores = (sim_scores(d, df_current, current_week, 15, semi_teams, pos_df, ros)
                          .set_index('team')
                          .reset_index()
                          .rename(columns={0:'score'}))

            semi_1 = semi_scores.iloc[[0,3]]
            semi_2 = semi_scores.iloc[[1,2]]

            # get finals matchup
            final_1 = semi_1.sort_values(by='score', ascending=False).iloc[0,0]
            final_2 = semi_2.sort_values(by='score', ascending=False).iloc[0,0]
            finals = [final_1, final_2]

            # count finals appearances
            for team in finals:
                n_finals[team] +=1

            third_1 = semi_1.sort_values(by='score').iloc[0,0]
            third_2 = semi_2.sort_values(by='score').iloc[0,0]
            third = [third_1, third_2]
            finals_teams = finals + third

            # simulate 2 weeks of finals matchups
            final_scores = (sim_scores(d, df_current, current_week, 16, finals_teams, pos_df, ros)
                           .set_index('team')
                           .merge(sim_scores(d, df_current, current_week, 17, finals_teams, pos_df, ros)
                           .set_index('team'), on='team')
                           .sum(axis=1)
                           .reset_index()
                           .rename(columns={0:'score'}))

            champ = final_scores[final_scores.team.isin(finals)].sort_values(by='score', ascending=False).iloc[0,0]

            # count championships and runner up
            n_champ[champ] += 1

            runner = final_scores[final_scores.team.isin(finals)].sort_values(by='score').iloc[0,0]
            n_second[runner] += 1

            third_pl = final_scores[final_scores.team.isin(third)].sort_values(by='score', ascending=False).iloc[0,0]

            # count third place
            n_third[third_pl] += 1

        # calculate averages and standard deviations
        avg = df.groupby('team').mean().reset_index().rename(columns={'wins':'avg_w', 'score':'avg_pts', 'rank':"avg_rnk"})
        #round(df.groupby('team')[['wins', 'score', 'rank']].agg(iqr), 1)
        sd = df.groupby('team').std().reset_index().rename(columns={'wins':'sd_w', 'score':'sd_pts', 'rank':'sd_rnk'})
        df = pd.merge(avg, sd, how='left', on='team').set_index('team')
        df = df.sort_values(by=['avg_w', 'avg_pts'], ascending=False)

        # get playoff table
        # convert dictionary counts to dataframes and combine
        playoffs = pd.DataFrame(n_playoffs.items(), columns=['team', 'n_playoffs'])
        finals = pd.DataFrame(n_finals.items(), columns=['team', 'n_finals'])
        champs = pd.DataFrame(n_champ.items(), columns=['team', 'n_champ'])
        runners = pd.DataFrame(n_second.items(), columns=['team', 'n_second'])
        thirds = pd.DataFrame(n_third.items(), columns=['team', 'n_third'])

        dfs = [playoffs, finals, champs, runners, thirds]

        playoff_sim = reduce(lambda left, right: pd.merge(left, right, on='team'), dfs)
        playoff_sim = playoff_sim.set_index('team')

        return table, df, rank_table, playoff_sim

    else:
        standings = sim_matchups(d, df_current, current_week, teams, pos_df, regular_season_end, matchup_week, matchups, ros)

        return standings

def scenarios(d, pr):
    '''
    Analyzes how a team would have performed in 3 scenarios:
        1: record vs every team every week
        2: record if team had another team's schedule
        3: record vs each team (1 matchup per week, loop for every team)
    '''

    regular_season_end = get_params(d)["regular_season_end"]
    df = get_params(d)["matchup_df"]
    teams = get_params(d)["teams"]
    teams = [x[:-1] for x in teams]
    teams = [x[:4] for x in teams]

    # set up data
    df['week'] = df['week'].astype(str).astype(int)
    df['score1'] = df['score1'].astype(str).astype(float)
    df['score2'] = df['score2'].astype(str).astype(float)
    df = df[(df.week <= regular_season_end) & (df.score1 > 0)]
    df["team1"] = df.team1.str[:-1].str[:4]
    df["team2"] = df.team2.str[:-1].str[:4]

    ### 1. Calculate record vs every team
    wins_vs_league = pr[['team', 'week', 'score']]
    wins_vs_league['teams_beat'] = wins_vs_league.groupby('week')['score'].rank() - 1

    ### 2. Calculate record given another team's schedule
    # set up matrix
    switched_sched = (pd.DataFrame(index=teams,
                               columns=teams)
                               .fillna(0))

    # set up dataframe to return team scores
    hm = df[['week', 'team1', 'score1']].rename(columns={'team1':'team', 'score1':'score'})
    aw = df[['week', 'team2', 'score2']].rename(columns={'team2':'team', 'score2':'score'})
    return_score = hm.append(aw)

    # for each team in list (team 1), go through every other team (team 2)
    # and replace team 1 score with team 2
    # if opponent in a week is the same as team 2, keep original schedule
    for tm1 in teams:
        #tm1 = 'bron'
        # return team 1's schedule
        hm = df[df.team1 == tm1]
        aw = df[df.team2 == tm1]
        aw = aw.rename(columns={"team1":"team2",
                          "score1":"score2",
                          "team2":"team1",
                          "score2":"score1"})
        sched = hm.append(aw).sort_values("week")

        # get team 2's score for the current week to replace with team 1
        for tm2 in teams:
            for wk in sched.week:
                # return replacement team score
                sched_slice = sched[(sched.week==wk)]
                score = return_score[(return_score.week==wk) & (return_score.team==tm2)]

                # replace scores
                # if current week's schedule remains the same, return record; otherwise repace scores
                if sched_slice.team2.values == tm2:
                    result = np.where(sched_slice.score2 > sched_slice.score1, 1, 0)
                else:
                    sched_slice['team1'] = tm2
                    sched_slice['score1'] = score.score.values

                # get result of hypothetica matchup if team 1 does not equal team 2
                if sched_slice.team2.values != tm2:
                    result = np.where(sched_slice.score1 > sched_slice.score2, 1, 0)

                switched_sched.loc[tm1, tm2] += result


    ### 3: record vs each opponent
    wins_vs_opp = (pd.DataFrame(index=teams,
                               columns=teams)
                               .fillna(0))

    hm = df[['week', 'team1', 'score1']].rename(columns={'team1':'team', 'score1':'score'})
    aw = df[['week', 'team2', 'score2']].rename(columns={'team2':'team', 'score2':'score'})
    return_score = hm.append(aw)

    # for each team (team 1), find how many times they would have beaten team 2
    for tm1 in teams:
        #tm1 = 'bron'
        # return team 1 scores
        hm = df[df.team1 == tm1].iloc[:,0:3]
        aw = df[df.team2 == tm1].iloc[:,[0,3,4]]
        aw = aw.rename(columns={"team1":"team2",
                          "score1":"score2",
                          "team2":"team1",
                          "score2":"score1"})
        sched = hm.append(aw).sort_values("week")

        teams2 = [x for x in teams if x != tm1]
        for tm2 in teams2:
            #tm2 = "gupt"
            # return team 2 scores
            hm2 = df[df.team1 == tm2].iloc[:,0:3]
            hm2 = hm2.rename(columns={"team1":"team2",
                      "score1":"score2"})
            aw2 = df[df.team2 == tm2].iloc[:,[0,3,4]]
            sched2 = hm2.append(aw2).sort_values("week")

            # combine schedules and find result for team 1
            matchups = pd.merge(sched, sched2, how="inner", on="week")
            wins = sum(np.where(matchups.score1 > matchups.score2, 1, 0))

            wins_vs_opp.loc[tm1, tm2] += wins

    return wins_vs_league, switched_sched, wins_vs_opp

# %% Lineup Efficiency
# code thanks again to steven morse

def get_matchups(league_id, season, week, swid='', espn=''):
    '''
    Pull full JSON of matchup data from ESPN API for a particular week.
    '''

    url = 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/' + str(season) + '/segments/0/leagues/' + str(league_id)

    r = requests.get(url + '?view=mMatchup&view=mMatchupScore',
                     params={'scoringPeriodId': week, 'matchupPeriodId': week},
                     cookies={"SWID": swid, "espn_s2": espn})

    d = r.json()

def opt_points(league_id, season, week, swid='', espn=''):
    '''
    Pull full JSON of matchup data from ESPN API for a particular week.
    '''

    url = 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/' + str(season) + '/segments/0/leagues/' + str(league_id)
    r = requests.get(url + '?view=mMatchup&view=mMatchupScore',
                     params={'scoringPeriodId': week, 'matchupPeriodId': week},
                     cookies={"SWID": swid, "espn_s2": espn})

    d = r.json()

    '''
    Constructs week team slates with slotted position,
    position, and points (actual and ESPN projected),
    given full matchup info (`get_matchups`)
    '''
    slotcodes = {
        0: 'QB', 1: 'QB',
        2: 'RB', 3: 'Flex',
        4: 'WR', 5: 'Flex',
        6: 'TE',
        10: 'LB', 11: 'DL',
        14: 'DB',
        16: 'DST', 17: 'K',
        18: 'P', 19: 'HC',
        20: 'Bench', 21: 'IR',
        23: 'Flex'}

    slates = {}
    slots = []
    for team in d['teams']:
        slate = []
        for p in team['roster']['entries']:
            # get name
            name  = p['playerPoolEntry']['player']['fullName']

            # get actual lineup slot
            slotid = p['lineupSlotId']
            slot = slotcodes[slotid]
            slots.append(slotid)
            # get projected and actual scores
            act, proj = 0, 0
            for stat in p['playerPoolEntry']['player']['stats']:
                if stat['scoringPeriodId'] != week:
                    continue
                if stat['statSourceId'] == 0:
                    act = stat['appliedTotal']
                elif stat['statSourceId'] == 1:
                    proj = stat['appliedTotal']
                else:
                    print('Error')

            # get type of player
            pos = 'Unk'
            ess = p['playerPoolEntry']['player']['eligibleSlots']
            if 0 in ess: pos = 'QB'
            elif 2 in ess: pos = 'RB'
            elif 4 in ess: pos = 'WR'
            elif 6 in ess: pos = 'TE'
            elif 10 in ess: pos = 'LB'
            elif 11 in ess: pos = 'DL'
            elif 14 in ess: pos = 'DB'
            elif 16 in ess: pos = 'DST'
            elif 17 in ess: pos = 'K'
            elif 18 in ess: pos = 'P'
            elif 19 in ess: pos = 'HC'

            slate.append([name, slotid, slot, pos, act, proj])

        slate = pd.DataFrame(slate, columns=['Name', 'SlotID', 'Slot', 'Pos', 'Actual', 'Proj'])
        slates[team['id']] = slate

    slots = list(dict.fromkeys(slots))

    '''
    Given slates, compute total roster pts:
    actual, optimal, and using ESPN projections

    Parameters
    --------------
    slates : `dict` of `DataFrames`
        (from `get_slates`)
    posns : `list`
        roster positions, e.g. ['QB','RB', 'WR', 'TE']
    struc : `list`
        slots per position, e.g. [1,2,2,1]

    * This is not flexible enough to handle "weird" leagues
    like 6 Flex slots with constraints on # total RB/WR

    Returns
    --------------
    `dict` of `dict`s with actual, ESPN, optimal points
    '''

    get_posns = load_data(league_id, season, swid, espn)
    regular_season_end = get_params(get_posns)["regular_season_end"]
    lineup_slots_df = get_params(get_posns)["lineup_slots_df"]
    pos_df = lineup_slots_df[(lineup_slots_df.posID != 20) & (lineup_slots_df.posID != 21)]
    posns = pos_df.pos.to_list()
    struc = pos_df.limit.to_list()

    data = {}

    for tmid, slate in slates.items():
        slate = slates[tmid]
        # go through each team roster
        pts = {'opts':0, 'apts':0,
               'QB.opts':0, 'QB.apts':0,
               'RB.opts':0, 'RB.apts':0,
               'WR.opts':0, 'WR.apts':0,
               'TE.opts':0, 'TE.apts':0,
               'Flex.opts':0, 'Flex.apts':0,
               'DST.opts':0, 'DST.apts':0,
               'DL.opts':0, 'DL.apts':0,
               'LB.opts':0, 'LB.apts':0,
               'DB.opts':0, 'DB.apts':0,
               'HC.opts':0, 'HC.apts':0,
               'P.opts':0, 'P.apts':0,
               'K.opts':0, 'K.apts':0}

        # Total actual points - starters
        pts['apts'] = slate.query('Slot not in ["Bench", "IR"]').filter(['Actual']).sum().values[0]

        # get starters and remove 3rd RB/WR or 2nd TE
        starters = slate.query('Slot != "Bench"')
        flpos = ["RB", "WR", "TE"]
        flnum = [2, 2, 1]
        f_st = []
        for pos, num in zip(flpos, flnum):
            player = starters.query('Pos == @pos').sort_values(by="Actual", ascending=False).groupby("Pos").nth(num).Name.tolist()
            f_st.append(player)
        f_st = [e for e in f_st if e != []]
        f_st = [i for s in f_st for i in s]

        starters = starters[~starters.Name.isin(f_st)]
        starters = starters.Name.tolist()


        # Total actual points - by position
        for pos in posns:
            pts[pos+'.apts'] += slate.query('Slot == @pos').filter(['Actual']).sum().values[0]

        # get flex position IDs
        #flex_pos = [i for i in [3,5,23] if i in slots]
        flex_pl = pd.DataFrame()
        # Optimal points
        #actflex = -100  # actual pts scored by flex

        for pos, num in zip(posns, struc):
            # actual points, sorted by actual outcome
            #pos = "WR"
            #num = 2
            t = slate.query('Pos == @pos').sort_values(by='Actual', ascending=False).filter(['Actual']).values[:,0]

            # sum up points
            pts['opts'] += t[:num].sum()  # total optimal points
            pts[pos+'.opts'] = t[:num].sum()  # position optimal points

            # get best flex plays
            if 'Flex' in posns:
                fnum = posns.count('Flex')
                if pos in ['RB', 'WR', 'TE'] and len(t) > num:
                    flex = slate.query('Pos == @pos').sort_values(by='Actual', ascending=False)[fnum:]
                    flex_pl = flex_pl.append(flex)
        flex_pl = flex_pl[~flex_pl.Name.isin(starters)]
        flex_opt = flex_pl.sort_values(by='Actual', ascending=False).filter(['Actual']).values[:fnum].sum()
        pts['Flex.opts'] = flex_opt

        # Add flex points to total optimal and position optimal
        pts['opts'] += flex_opt

        data[tmid] = pts


    team_map = get_params(get_posns)["team_map"]
    df = pd.DataFrame.from_dict(data, orient='index')
    df['week'] = week
    df['team'] = df.index
    df = df.replace({'team':team_map})

    return df

def get_optimal(d, league_id, season, swid, espn):
    df_scores = pd.DataFrame()
    #regular_season_end = get_params(d)["regular_season_end"]+1
    matchup_week = get_params(d)["matchup_week"]
    for week in range(1, matchup_week):
        df = opt_points(league_id, season, week, swid, espn)
        df_scores = df_scores.append(df)

    return df_scores
