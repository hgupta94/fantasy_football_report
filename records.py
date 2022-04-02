import pandas as pd
import numpy as np
from all_functions import *
from league_analyzer import *
import json

####################
# 1. Most wins
# 2. Fewest wins
# 3. Most points - season
# 4. Fewest points - season
# 5. Most points - week
# 6. Fewest points - week
# 7. Fewest points in a win
# 8. Most points in a loss
# 9. Closest win
# 10. Biggest bloweout
# 11. Most points in a matchup
# 13. Fewest points in a matchup
# 13. Best point differential
# 14. Worst point differential
####################

# load data pr data
league_id = 1382012
season = 2021
swid = "{E01C2393-2E6F-420B-9C23-932E6F720B61}"
espn = "AEAVE3tAjA%2B4WQ04t%2FOYl15Ye5f640g8AHGEycf002gEwr1Q640iAvRF%2BRYFiNw5T8GSED%2FIG9HYOx7iwYegtyVzOeY%2BDhSYCOJrCGevkDgBrhG5EhXMnmiO2GpeTbrmtHmFZAsao0nYaxiKRvfYNEVuxrCHWYewD3tKFa923lw3NC8v5qjjtljN%2BkwFXSkj91k2wxBjrdaL5Pp1Y77%2FDzQza4%2BpyJq225y4AUPNB%2FCKOXYF7DTZ5B%2BbuHfyUKImvLaNJUTpwVXR74dk2VUMD9St"
d = load_data(league_id, season, swid, espn)
pr_all = power_table(league_id, season, swid, espn)[0]

##### load matchups data
matchups_all = pd.DataFrame()
for ssn in range(2018, season + 1):
    d = load_data(league_id, ssn, swid, espn)
    matchups = get_params(d)["matchup_df"]
    matchups["season"] = ssn
    matchups_all = matchups_all.append(matchups)

records = pd.DataFrame(columns=["cat", "record", "holder", "season", "week"])

matchups_all["team1"] = matchups_all["team1"]
matchups_all["team2"] = matchups_all["team2"]
matchups_all["score_diff"] = abs(matchups_all.score1 - matchups_all.score2)
matchups_all["tot_score"] = abs(matchups_all.score1 + matchups_all.score2)

# 1. Most wins
wins = pr_all.groupby(["season", "team"])[["result"]].sum().reset_index()
wins["season"] = wins.season.astype(str)
most_wins = wins[wins.result == wins.result.max()].rename(columns={"result": "record", "team": "holder"})
teams = most_wins.groupby('record')['holder'].apply(', '.join).reset_index()
seasons = most_wins.groupby('record')['season'].apply(', '.join).reset_index()
most_wins = pd.merge(teams, seasons, on="record")
most_wins["cat"] = "Most Wins"
most_wins["record"] = most_wins["record"].astype(int).astype(str)
records = records.append(most_wins)

# 2. Most losses
pr_all["loss"] = np.where(pr_all.result == 0, 1, 0)
losses = pr_all.groupby(["season", "team"])[["loss"]].sum().reset_index()
losses["season"] = losses.season.astype(str)
most_losses = losses[losses.loss == losses.loss.max()].rename(columns={"loss": "record", "team": "holder"})
teams = most_losses.groupby('record')['holder'].apply(', '.join).reset_index()
seasons = most_losses[["season", "record"]].groupby('record')['season'].apply(', '.join).reset_index()
least_wins = pd.merge(teams, seasons, on="record")
least_wins["cat"] = "Most Losses"
least_wins["record"] = least_wins["record"].astype(int).astype(str)
records = records.append(least_wins)

# 3. Most points - season
pts_ssn = pr_all.groupby(["season", "team"])[["score"]].sum().reset_index()
weeks = pr_all.groupby(["season"])[["week"]].max().reset_index()
pts_ssn = pts_ssn.merge(weeks, on="season")
pts_ssn["score"] = pts_ssn.score / pts_ssn.week
pts_ssn = pts_ssn.drop(["week"], axis=1)
most_pts_ssn = pts_ssn[pts_ssn.score == pts_ssn.score.max()].rename(columns={"score": "record", "team": "holder"})
most_pts_ssn["cat"] = "Highest PPG (S)"
most_pts_ssn["record"] = most_pts_ssn["record"].round(2).astype(str)
records = records.append(most_pts_ssn)

# 11. Most points in a matchup
matchup_high = matchups_all[matchups_all.tot_score == matchups_all.tot_score.max()]
matchup_high["holder"] = matchup_high.team1 + " (" + matchup_high.score1.astype(
    str) + ") - " + matchup_high.team2 + " (" + matchup_high.score2.astype(str) + ")"
matchup_high = matchup_high[["week", "tot_score", "holder", "season"]].rename(columns={"tot_score": "record"})
matchup_high["cat"] = "Most Points (M)"
matchup_high["record"] = matchup_high["record"].round(2).astype(str)
records = records.append(matchup_high)

# 5. most points - week
most_pts_wk = pr_all[pr_all.score == pr_all.score.max()][["team", "score", "week", "season"]].rename(
    columns={"score": "record", "team": "holder"})
most_pts_wk["cat"] = "Most Points (W)"
most_pts_wk["record"] = most_pts_wk["record"].round(2).astype(str)
records = records.append(most_pts_wk)

# 4. Fewest points - season
least_pts_ssn = pts_ssn[pts_ssn.score == pts_ssn.score.min()].rename(columns={"score": "record", "team": "holder"})
least_pts_ssn["cat"] = "Fewest PPG (S)"
least_pts_ssn["record"] = least_pts_ssn["record"].round(2).astype(str)
records = records.append(least_pts_ssn)

# 12. Fewest points in a matchup
matchup_low = matchups_all[matchups_all.tot_score == matchups_all.tot_score.min()]
matchup_low["holder"] = matchup_low.team1 + " (" + matchup_low.score1.astype(
    str) + ") - " + matchup_low.team2 + " (" + matchup_low.score2.astype(str) + ")"
matchup_low = matchup_low[["week", "tot_score", "holder", "season"]].rename(columns={"tot_score": "record"})
matchup_low["cat"] = "Fewest Points (M)"
matchup_low["record"] = matchup_low["record"].round(2).astype(str)
records = records.append(matchup_low)

# 6. fewest points - week
least_pts_wk = pr_all[pr_all.score == pr_all.score.min()][["team", "score", "week", "season"]].rename(
    columns={"score": "record", "team": "holder"})
least_pts_wk["cat"] = "Fewest Points (W)"
least_pts_wk["record"] = least_pts_wk["record"].round(2).astype(str)
records = records.append(least_pts_wk)

# 7. most points in a loss
most_pts_l = pr_all[pr_all.result == 0]
most_pts_l = most_pts_l[most_pts_l.score == most_pts_l.score.max()][["team", "score", "week", "season"]].rename(
    columns={"score": "record", "team": "holder"})
most_pts_l["cat"] = "Most Points in Loss"
most_pts_l["record"] = most_pts_l["record"].round(2).astype(str)
records = records.append(most_pts_l)

# 8. fewest points in a win
least_pts_w = pr_all[pr_all.result == 1]
least_pts_w = least_pts_w[least_pts_w.score == least_pts_w.score.min()][["team", "score", "week", "season"]].rename(
    columns={"score": "record", "team": "holder"})
least_pts_w["cat"] = "Fewest Points in Win"
least_pts_w["record"] = least_pts_w["record"].round(2).astype(str)
records = records.append(least_pts_w)

# 9. Closest matchup
closest = matchups_all[matchups_all.score_diff == matchups_all.score_diff.min()]
closest["holder"] = closest.team1 + " (" + closest.score1.astype(
    str) + ")-" + closest.team2 + " (" + closest.score2.astype(str) + ")"
closest = closest[["week", "score_diff", "holder", "season"]].rename(columns={"score_diff": "record"})
closest["cat"] = "Closest Matchup"
closest["record"] = closest["record"].round(2).astype(str)
records = records.append(closest)

# 10. Biggest blowout
blowout = matchups_all[matchups_all.score_diff == matchups_all.score_diff.max()]
blowout["holder"] = blowout.team1 + " (" + blowout.score1.astype(
    str) + ")-" + blowout.team2 + " (" + blowout.score2.astype(str) + ")"
blowout = blowout[["week", "score_diff", "holder", "season"]].rename(columns={"score_diff": "record"})
blowout["cat"] = "Biggest Blowout"
blowout["record"] = blowout["record"].round(2).astype(str)
records = records.append(blowout)

records = records.replace(np.nan, "").reset_index(drop=True)
records = records.rename(
    columns={"cat": "Category", "record": "Record", "holder": "Holder", "season": "Season", "week": "Week"})
headings_rec = tuple(records.columns)
data_rec = [tuple(x) for x in records.to_numpy()]
