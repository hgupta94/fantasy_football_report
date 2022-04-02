from all_functions import *
import pandas as pd
import numpy as np
import datetime as dt

print("Simulating the season...")
s = dt.datetime.now()

today = dt.datetime.today()
weekday = today.weekday()
hour = today.hour
folder = "//home//hgupta//fantasy-football-league-report//simulation_tables//"

# only run simulation on wednesday mornings (takes too long)
if (weekday == 2) & (hour < 10):
    # load data for simulation
    league_id = 1382012
    season = 2021
    swid = "{E01C2393-2E6F-420B-9C23-932E6F720B61}"
    espn = "AEAVE3tAjA%2B4WQ04t%2FOYl15Ye5f640g8AHGEycf002gEwr1Q640iAvRF%2BRYFiNw5T8GSED%2FIG9HYOx7iwYegtyVzOeY%2BDhSYCOJrCGevkDgBrhG5EhXMnmiO2GpeTbrmtHmFZAsao0nYaxiKRvfYNEVuxrCHWYewD3tKFa923lw3NC8v5qjjtljN%2BkwFXSkj91k2wxBjrdaL5Pp1Y77%2FDzQza4%2BpyJq225y4AUPNB%2FCKOXYF7DTZ5B%2BbuHfyUKImvLaNJUTpwVXR74dk2VUMD9St"
    d = load_data(league_id, season, swid, espn)

    regular_season_end = get_params(d)["regular_season_end"]
    matchup_week = get_params(d)["matchup_week"]
    teams = get_params(d)["teams"]
    matchups = get_params(d)["matchup_df"].reset_index(drop=True)
    sim_week = d['scoringPeriodId']
    current_week = get_params(d)["current_week"]
    n_teams = get_params(d)["playoff_teams"]
    positions = get_params(d)["positions"]
    league_size = get_params(d)["league_size"]
    team_map =  get_params(d)["team_map"]
    slotcodes = get_params(d)["slotcodes"]
    roster_size = get_params(d)["roster_size"]
    lineup_slots_df = get_params(d)["lineup_slots_df"]
    pos_df = lineup_slots_df[(lineup_slots_df.posID != 20) & (lineup_slots_df.posID != 21)]
    ros = get_ros_proj(d, current_week, matchup_week, league_size, roster_size, team_map)
    team_projections = ros[0]
    player_projections = ros[1]
    df_current = power_rank(d, league_id, season)

    # ~9s per season sim
    n_sim = 1000
    season_sim = sim_season(d, league_id, season, current_week, matchup_week, regular_season_end, df_current, teams, matchups, ros, n_sim=n_sim)
    w_table = season_sim[0].reset_index()
    s_table = season_sim[1].reset_index()
    r_table = season_sim[2].reset_index()
    p_table = season_sim[3].reset_index()

    # summary table
    s_table[["avg_pts", "sd_pts"]] = s_table[["avg_pts", "sd_pts"]] / regular_season_end
    s_table["sd_pts"] = s_table.sd_pts * 3      # 3 standard devations
    s_table[["avg_w", "sd_w", "avg_pts", "sd_pts", "avg_rnk", "sd_rnk"]] = round(s_table[["avg_w", "sd_w", "avg_pts", "sd_pts", "avg_rnk", "sd_rnk"]], 1)
    s_table = s_table[["team", "avg_w", "sd_w", "avg_pts", "sd_pts", "avg_rnk", "sd_rnk"]]
    s_table["team"] = s_table.team.str[:-1].str[:4]
    s_table.rename(columns={'team':'Team',
                                  'avg_w':'Wins',
                                  'sd_w': '+/-',
                                  'avg_pts':'PPG',
                                  'sd_pts': '+/-',
                                  'avg_rnk': 'Rank',
                                  'sd_rnk': '+/-'}, inplace=True)
    s_table.to_csv(folder + "summary_table.csv", index=False)

    # get order of teams for wins and ranks tables
    rank_order = list(s_table.sort_values("Rank").Team)
    wins_order = list(s_table.sort_values("Wins", ascending=False).Team)

    # wins distribution table
    w_format = w_table.iloc[:, 1:regular_season_end+2] / n_sim
    w_format = (w_format*100).round(0).astype(int).astype(str) + "%"
    w_format.replace("0%", "", inplace=True)
    w_table.iloc[:, 1:regular_season_end+2] = w_format
    w_table.rename(columns={'index':'Team', 'avg_w':'xWins'}, inplace=True)
    w_table["Team"] = w_table.Team.str[:-1].str[:4]
    true_sort_w = [s for s in wins_order if s in w_table.Team.unique()]
    w_table = w_table.set_index('Team').loc[true_sort_w].reset_index()
    w_table.to_csv(folder + "wins_table.csv", index=False)

    # rank distribution
    r_format = r_table.iloc[:, 1:len(teams)+1] / n_sim
    r_format = (r_format*100).round(0).astype(int).astype(str) + "%"
    r_format.replace("0%", "", inplace=True)
    r_table.iloc[:, 1:len(teams)+1] = r_format
    r_table.rename(columns={'index':'Team', 'avg_rnk':'xRank'}, inplace=True)
    r_table["Team"] = r_table.Team.str[:-1].str[:4]
    true_sort_r = [s for s in rank_order if s in r_table.Team.unique()]
    r_table = r_table.set_index('Team').loc[true_sort_r].reset_index()
    r_table.to_csv(folder + "ranks_table.csv", index=False)

    # playoff probabilities
    p_table['xPay'] = ((p_table.n_champ) / n_sim * 500) + ((p_table.n_second) / n_sim * 250) + ((p_table.n_third) / n_sim * 100)
    p_table = p_table.sort_values(['xPay', 'n_playoffs'], ascending=[False, False])
    p_format = p_table.iloc[:, 1:6] / n_sim
    p_format = round(p_format*100, 1).astype(int).astype(str) + "%"
    p_table.iloc[:, 1:6] = p_format
    p_table['xPay'] = "$" + p_table['xPay'].astype(int).astype(str)
    p_table["team"] = p_table.team.str[:-1].str[:4]
    p_table.rename(columns={'team':'Team', 'n_playoffs':'Playoffs', 'n_finals':'Finals', 'n_champ':'1st', 'n_second':'2nd', 'n_third':'3rd', 'xPay': 'Pay'}, inplace=True)
    p_table.to_csv(folder + "playoffs_table.csv", index=False)

    # save summary table every wednesday for future reference
    today = dt.date.today()
    weekday = today.weekday()
    timestamp = today.strftime('%Y') + today.strftime('%m') + today.strftime('%d')

    file = "season_sim_results" + timestamp + ".csv"
    s_table.to_csv(folder + "weekly_summary//" + file, index=False)
else:
    # otherwise load CSVs
    s_table = pd.read_csv(folder + "summary_table.csv").rename(columns={"+/-.1":"+/-", "+/-.2":"+/-"})
    w_table = pd.read_csv(folder + "wins_table.csv", keep_default_na=False)
    r_table = pd.read_csv(folder + "ranks_table.csv", keep_default_na=False)
    p_table = pd.read_csv(folder + "playoffs_table.csv")


# timestamp for table caption
today = dt.date.today()
offset = (today.weekday() - 2) % 7
last_wed = today - dt.timedelta(days=offset)
tstamp_ssn = "Updated: " + str(last_wed.strftime("%A %b %d, %Y"))


# for rendering to page
headings_s = tuple(s_table.columns)
data_s = [tuple(x) for x in s_table.to_numpy()]

headings_w = tuple(w_table.columns)
data_w = [tuple(x) for x in w_table.to_numpy()]

headings_r = tuple(r_table.columns)
data_r = [tuple(x) for x in r_table.to_numpy()]

headings_p = tuple(p_table.columns)
data_p = [tuple(x) for x in p_table.to_numpy()]

print("Season simulation complete! Time elapsed:")
e = dt.datetime.now()
el = e - s
print(el)
