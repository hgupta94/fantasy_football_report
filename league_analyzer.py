import os
import pandas as pd
import numpy as np
from all_functions import load_data, get_params, power_rank, scenarios, get_optimal
import json
from flask import request
import requests
from plotnine import *
import adjustText

#%% Power Rankings
# Load power rankings for each eligible year
def power_table(league_id, season, swid, espn):
    pr = pd.DataFrame()
    for yr in range(2018, season+1):
        # load data if league existed
        d = load_data(league_id, yr, swid, espn)
        if 'messages' in d:
            pass
        else:
            regular_season_end = get_params(d)["regular_season_end"]
            df = power_rank(d, league_id, yr)
            df['season'] = yr
            pr = pr.append(df)

    # Get table
    last_week = get_params(d)["current_week"]
    last_week = int(np.where(last_week > regular_season_end, regular_season_end, last_week))
    pr_table = pr[(pr.season == int(season)) & (pr.week == last_week)]
    pr_table["power_score"] = round(pr_table.power_score)
    pr_table = pr_table[["team", "power_rank", "rank_change", "power_score", "score_change"]].sort_values(["power_rank"])
    pr_table.iloc[:,[1,2,3,4]] = pr_table.iloc[:,[1,2,3,4]].astype(int)
    pr_table.iloc[:,[2,4]] = pr_table.iloc[:,[2,4]].applymap(lambda x: "+"+str(x) if x>0 else x)

    return pr, pr_table

#%% Simulations


#%% Scenarios
def get_wins_vs_lg(d, wins_vs_lg, pr, ssn):
    # Get record vs league for each team

    # set parameters
    teams = get_params(d)["league_size"]
    team_names = get_params(d)["teams"]
    team_names = [x[:-1] for x in team_names]
    team_names = [x[:4] for x in team_names]
    last_week = get_params(d)["current_week"]
    regular_season_end = get_params(d)["regular_season_end"]
    weeks_played = np.where(last_week > regular_season_end, regular_season_end, last_week).item()

    tot_wins = wins_vs_lg.groupby(["team"]).sum("teams_beat").rename(columns={"teams_beat":"wins"})
    tot_wins["wins"] = tot_wins.wins.astype("int")
    tot_wins["losses"] = (weeks_played*(teams-1)) - tot_wins.wins
    tot_wins["win_perc"] = round(tot_wins["wins"] / (last_week*(teams-1)), 3)
    tot_wins["win_perc"] = tot_wins.win_perc.map('{:.3f}'.format)
    tot_wins = tot_wins.sort_values(["win_perc"], ascending=False)
    tot_wins["record"] = tot_wins.wins.apply(str) + "-" + tot_wins.losses.apply(str)

    # Get record vs each team
    wins_vs_opp = scenarios(d, pr[pr.season == int(ssn)])[2]
    wins_vs_opp.reset_index(level=0, inplace=True)

    # convert to long go get record vs each team
    v_vars = list(wins_vs_opp.columns[1:teams+1])
    long = pd.melt(wins_vs_opp, id_vars=["index"], value_vars=v_vars).rename(columns={"value":"wins", "index":"team"})
    long['losses'] = weeks_played - long.wins
    long['record'] = long.wins.apply(str) + "-" + long.losses.apply(str)
    long['record'] = np.where(long.team == long.variable, "", long.record)

    # convert back to wide and add record/win_perc
    wins_vs_opp_new = long.pivot(index="team", columns="variable", values="record")
    wins_vs_opp_new = wins_vs_opp_new.merge(tot_wins.loc[:,["record", "win_perc"]], left_index=True, right_index=True).sort_values(["win_perc"], ascending=False)
    cols = wins_vs_opp_new.index.tolist()
    cols.extend(["record", "win_perc"])
    wins_vs_opp_new = wins_vs_opp_new[cols]
    wins_vs_opp_new['team'] = wins_vs_opp_new.index

    # reoder columns so "teams" is first
    cols = list(wins_vs_opp_new.columns)
    cols = [cols[-1]] + cols[:-1]
    wins_vs_opp_new = wins_vs_opp_new[cols]
    wins_vs_opp_new = wins_vs_opp_new.rename(columns={"record":"W/L", "win_perc":"Win%"})

    return wins_vs_opp_new

def get_sched_switch(d, sched_switch):
    sched_switch.reset_index(level=0, inplace=True)

    # set parameters
    teams = get_params(d)["league_size"]
    team_names = get_params(d)["teams"]
    team_names = [x[:-1] for x in team_names]
    team_names = [x[:4] for x in team_names]
    last_week = get_params(d)["current_week"]
    regular_season_end = get_params(d)["regular_season_end"]
    weeks_played = np.where(last_week > regular_season_end, regular_season_end, last_week).item()

    # convert to long to return record
    v_vars = list(sched_switch.columns[1:teams+1])
    long = pd.melt(sched_switch, id_vars=["index"], value_vars=v_vars).rename(columns={"value":"wins", "index":"team"})
    long['losses'] = weeks_played - long.wins
    long['record'] = long.wins.apply(str) + "-" + long.losses.apply(str)

    # convert back to wide
    sched_switch_new = long.pivot(index="variable", columns="team", values="record")

    # reorder teams to match previous table
    sched_switch_new = sched_switch_new.loc[:,team_names]
    sched_switch_new = sched_switch_new.reindex(team_names)
    sched_switch_new["team"] = sched_switch_new.index

    cols = list(sched_switch_new.columns)
    cols = [cols[-1]] + cols[:-1]
    sched_switch_new = sched_switch_new[cols]

    return sched_switch_new


#%% Team and position efficiencies
def get_eff(eff, pos_list):
    eff.columns = eff.columns.str.lower()
    #eff.columns = eff.columns.str.replace('\/', '', regex=True)

    eff = eff.groupby("team").sum()

    # calculate team and positional efficiencies
    eff['team.eff'] = (eff.apts / eff.opts)
    eff_dict = {}
    for pos in pos_list:
        # use inverse if act & opt are negative and opt>act (ie. act=-3 and opt=-1) - mainly for defense
        # otherwise efficiency will be over 100%
        if any((eff[pos+'.apts'] < 0) & (eff[pos+'.opts'] < 0) & (eff[pos+'.opts'] > eff[pos+'.apts'])):
            eff_dict[format(pos) + ".eff"] = eff[pos+'.opts'] / eff[pos+'.apts']
        else:
            eff_dict[format(pos) + ".eff"] = eff[pos+'.apts'] / eff[pos+'.opts']

    df = pd.DataFrame(eff_dict)

    # replace nan's with 1 because that is still the most efficient outcome (actual and optimal are 0)
    df = df.fillna(1)

    # replace inf with 0 (best player scored 0, but actual starter scored less)
    df.replace([np.inf, -np.inf], 0,inplace=True)

    # replace eff>1 with 1 (should be only flex - single player is compared to all eligible flex players)
    # and replace eff<0 with 0  (actual was negative)
    df[df > 1] = 1
    df[df < 0] = 0

    # combine with efficiency df and rename team opts and apts
    eff2 = pd.concat([eff, df], axis=1)
    eff2 = eff2.rename(columns={"opts":"team.opts", "apts":"team.apts"})

    # replace Flex efficiency with RB+WR+TE efficiency
    eff2['flex.apts'] = (eff2['rb.apts'] + eff2['wr.apts'] + eff2['te.apts'])
    eff2['flex.opts'] = (eff2['rb.opts'] + eff2['wr.opts'] + eff2['te.opts'])
    eff2['flex.eff'] = (eff2['rb.apts'] + eff2['wr.apts'] + eff2['te.apts']) / (eff2['rb.opts'] + eff2['wr.opts'] + eff2['te.opts'])

    # remove columns with all 0s
    eff2 = eff2.loc[:, (eff2 != 0).any(axis=0)]
    eff2 = eff2.reset_index()

    return eff2

# Efficiency by position
def team_efficiency(d, eff):
    regular_season_end = get_params(d)["regular_season_end"]
    current_week = get_params(d)["current_week"]
    matchup_week = get_params(d)["matchup_week"]
    week = np.where(matchup_week > regular_season_end, regular_season_end, current_week).item()

    # create summary variables to plot
    eff_pos = eff.loc[:,["team", "week", "team.opts", "team.apts"]]

    df = pd.DataFrame()
    df["team"] = eff_pos.iloc[:,0].values
    df["Position"] = eff_pos.iloc[:,2].values
    df["opt"] = eff_pos.iloc[:,2].values
    df["act"] = eff_pos.iloc[:,3].values
    df["diff"] = df["act"] - df["opt"]
    df["optPerWeek"] = df["opt"] / week
    df["actPerWeek"] = df["act"] / week
    df["diffPerWeek"] = df["actPerWeek"] - df["optPerWeek"]
    df["effic"] = df["actPerWeek"] / df["optPerWeek"]
    df["effic"] = df["effic"].round(2).apply(lambda x: format(x, '.0%'))

    return df

def pos_efficiency(eff):
    cols2 = ["team"] + [col for col in eff if col.endswith("eff")]
    eff_pos = eff.loc[:,cols2]
    eff_pos = eff_pos.groupby("team").agg("mean")
    eff_pos["team"] = eff_pos.index
    eff_pos_long = pd.melt(eff_pos, id_vars=["team"], value_vars=list(eff_pos.columns[:-1])).rename(columns={"value":"avg_eff"})
    eff_pos_long["Position"] = eff_pos_long['variable'].str.split('.').str[0]
    eff_pos_long["Position"] = eff_pos_long["Position"].str.upper()

    return eff_pos_long
