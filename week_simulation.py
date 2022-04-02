from all_functions import *
import pandas as pd
import numpy as np
import datetime as dt
import lxml
import os

print("Simulating current week...")
s = dt.datetime.now()

# load aggregate projections
league_id = 1382012
season = 2021
swid = "{E01C2393-2E6F-420B-9C23-932E6F720B61}"
espn = "AEAVE3tAjA%2B4WQ04t%2FOYl15Ye5f640g8AHGEycf002gEwr1Q640iAvRF%2BRYFiNw5T8GSED%2FIG9HYOx7iwYegtyVzOeY%2BDhSYCOJrCGevkDgBrhG5EhXMnmiO2GpeTbrmtHmFZAsao0nYaxiKRvfYNEVuxrCHWYewD3tKFa923lw3NC8v5qjjtljN%2BkwFXSkj91k2wxBjrdaL5Pp1Y77%2FDzQza4%2BpyJq225y4AUPNB%2FCKOXYF7DTZ5B%2BbuHfyUKImvLaNJUTpwVXR74dk2VUMD9St"
d = load_data(league_id, season, swid, espn)

week_bet = np.where(get_params(d)["matchup_week"] > get_params(d)["regular_season_end"],
                "Final " + str(season),
                "Week " + str(get_params(d)["matchup_week"]))
week_bet = week_bet.item(0)

today = dt.datetime.today()
weekday = today.weekday()
hour = today.hour
folder = "//home//hgupta//fantasy-football-league-report//simulation_tables//"

if (hour % 8 == 0):
    # run every 8 hours
    positions = get_params(d)["positions"]
    matchup_week = get_params(d)["matchup_week"]
    league_size = get_params(d)["league_size"]
    team_map =  get_params(d)["team_map"]
    matchups = get_params(d)["matchup_df"].reset_index(drop=True)
    teams = get_params(d)["teams"]
    slotcodes = get_params(d)["slotcodes"]
    lineup_slots_df = get_params(d)["lineup_slots_df"]
    pos_df = lineup_slots_df[(lineup_slots_df.posID != 20) & (lineup_slots_df.posID != 21)]
    faab = get_params(d)["faab_remaining"]
    faab["team"] = faab.team.str[:-1].str[:4]

    # takes ~0.5s per sim
    n_sim = 1000
    curr_wk_sim = sim_week(d, positions, matchup_week, league_size, team_map, matchups, teams, slotcodes, pos_df, n_sim=n_sim)
    avg_score = curr_wk_sim[1].groupby("team").mean().reset_index()
    avg_score["score"] = np.round(avg_score.score, 1)

    # get betting table
    betting_table = pd.merge(curr_wk_sim[0], avg_score, on="team")
    betting_table[["n_wins", "n_highest", "n_lowest"]] = betting_table[["n_wins", "n_highest", "n_lowest"]] / n_sim

    # calculate betting lines
    betting_table["game_line"] = np.round(np.where(betting_table.n_wins > 0.5,
                 (100 * betting_table.n_wins) / (1 - betting_table.n_wins) * -1,
                 100 / (betting_table.n_wins) - 100))
    betting_table["game_line"] = np.where(~np.isfinite(betting_table.game_line), 0, betting_table.game_line)
    betting_table["game_line"] = np.where(betting_table.game_line > 20000, 20000, betting_table.game_line)
    betting_table["game_line"] = betting_table.game_line.round(-1)

    betting_table["high_line"] = np.round(np.where(betting_table.n_highest > 0.5,
                 (100 * betting_table.n_highest) / (1 - betting_table.n_highest) * -1,
                 100 / (betting_table.n_highest) - 100))
    betting_table["high_line"] = np.where(~np.isfinite(betting_table.high_line), 0, betting_table.high_line)
    betting_table["high_line"] = np.where(betting_table.high_line > 20000, 20000, betting_table.high_line)
    betting_table["high_line"] = betting_table.high_line.round(-1)

    betting_table["low_line"] = np.round(np.where(betting_table.n_lowest > 0.5,
                 (100 * betting_table.n_lowest) / (1 - betting_table.n_lowest) * -1,
                 100 / (betting_table.n_lowest) - 100))
    betting_table["low_line"] = np.where(~np.isfinite(betting_table.low_line), 0, betting_table.low_line)
    betting_table["low_line"] = np.where(betting_table.low_line > 10000, 10000, betting_table.low_line)
    betting_table["low_line"] = betting_table.low_line.round(-1)

    betting_table.replace([np.inf, -np.inf], 0, inplace=True)

    betting_table = betting_table[["team", "game_id", "n_wins", "game_line", "score", "high_line", "low_line"]]
    betting_table = betting_table.sort_values(["game_id", "n_wins"])
    betting_table['n_wins'] = pd.Series(["{0:.0f}%".format(val * 100) for val in betting_table['n_wins']], index = betting_table.index)

    betting_table[["game_line", "high_line", "low_line"]] = betting_table[["game_line", "high_line", "low_line"]].astype(int).applymap(lambda x: "+"+str(x) if x>0 else x)

    betting_table = betting_table.drop("game_id", axis=1)
    betting_table["team"] = betting_table.team.str[:-1].str[:4]

    # add faab budgets
    betting_table = pd.merge(betting_table, faab, on="team", how="left")
    betting_table["faab_left"] = "$" + betting_table["faab_left"].astype(str)

    # reformat 0s
    betting_table.replace(0, "–", inplace=True)

    betting_table.columns = ["Team", "Win%", "Game Line", "Avg Score", "High Score", "Low Score", "FAAB"]

    # save table with timestamp
    if hour == 8:
        today = dt.datetime.today()
        timestamp = today.strftime('%Y') + today.strftime('%m') + today.strftime("%d") + today.strftime("%H")
        file = "betting_table_" + timestamp + ".csv"
        betting_table.to_csv("//home//hgupta//fantasy-football-league-report//betting_tables//" + file, index=False)

    # save table for quick load
    betting_table.to_csv(folder + "betting_table.csv", index=False)
else:
    # if ran outside normal time, load betting table
    betting_table = pd.read_csv(folder + "betting_table.csv", keep_default_na=False)
    betting_table.replace("–", "0", inplace=True)
    betting_table[["Game Line", "High Score", "Low Score"]] = betting_table[["Game Line", "High Score", "Low Score"]].astype(int)
    betting_table[["Game Line", "High Score", "Low Score"]] = betting_table[["Game Line", "High Score", "Low Score"]].applymap(lambda x: "+"+str(x) if x>0 else x)
    betting_table.replace(0, "–", inplace=True)

# timestamp for table caption
file = os.path.getmtime(folder + "betting_table.csv")
tstamp = dt.datetime.fromtimestamp(file) - dt.timedelta(hours=4)
#tstamp = dt.datetime.now() - dt.timedelta(hours=4)
tstamp = "Updated: " + str(tstamp.strftime("%A %b %d, %Y %I:%M %p"))

# for rendering to webpage
headings_bets = tuple(betting_table.columns)
data_bets = [tuple(x) for x in betting_table.to_numpy()]

print("Weekly simulation complete! Time elapsed:")
e = dt.datetime.now()
el = e - s
print(el)
