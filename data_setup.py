import pandas as pd
from plotnine import *
from all_functions import *
from league_analyzer import *
import io
import base64
import os
import datetime as dt
today = dt.date.today()
weekday = today.weekday()

##### load and set up data #####
league_id = 1382012
season = 2021
swid = "{E01C2393-2E6F-420B-9C23-932E6F720B61}"
espn = "AEAVE3tAjA%2B4WQ04t%2FOYl15Ye5f640g8AHGEycf002gEwr1Q640iAvRF%2BRYFiNw5T8GSED%2FIG9HYOx7iwYegtyVzOeY%2BDhSYCOJrCGevkDgBrhG5EhXMnmiO2GpeTbrmtHmFZAsao0nYaxiKRvfYNEVuxrCHWYewD3tKFa923lw3NC8v5qjjtljN%2BkwFXSkj91k2wxBjrdaL5Pp1Y77%2FDzQza4%2BpyJq225y4AUPNB%2FCKOXYF7DTZ5B%2BbuHfyUKImvLaNJUTpwVXR74dk2VUMD9St"

d = load_data(league_id, season, swid, espn)
week = np.where(get_params(d)["current_week"] > get_params(d)["regular_season_end"],
                "Final " + str(season),
                "Week " + str(get_params(d)["current_week"]))
week = week.item(0)


##### Standings #####
regular_season_end = get_params(d)["regular_season_end"]
current_week = get_params(d)["current_week"]
standings = power_rank(d, league_id, season)[["xwins", "team", "week", "wins", "total_pf"]]
standings = standings[standings.week == standings.week.max()]
standings["total_pf"] = standings.total_pf.round(2)
standings["losses"] = current_week - standings["wins"]
standings["xlosses"] = regular_season_end - standings["xwins"]
standings["xrecord"] = standings.xwins.round(1).astype(str) + "-" + standings.xlosses.round(1).astype(str)
standings["record"] = standings.wins.astype(int).astype(str) + "-" + standings.losses.astype(int).astype(str)
standings["win_perc"] = round(standings["wins"] / current_week, 3)
standings["win_perc"] = standings.win_perc.map('{:.3f}'.format)
standings["team"] = standings.team.str[:-1].str[:4]

# get top by by wins
top_five = standings.sort_values(["wins", "total_pf"], ascending=False).head(5)
top_five_list = top_five.team.tolist()

# get 6th place by points and append to top 5
sixth = standings[~standings.team.isin(top_five_list)].sort_values("total_pf", ascending=False).head(1)
final_standings = pd.concat([top_five, sixth])
playoff_list = final_standings.team.tolist()

# get bottom 4 by wins
bottom_four = standings[~standings.team.isin(playoff_list)].sort_values(["wins", "total_pf"], ascending=False)
final_standings = pd.concat([final_standings, bottom_four])

# calculate games back: bye, fifth seed; points back of 6th seed
last_bye_wins = final_standings.iloc[1,:].wins
final_standings["gb_bye"] = last_bye_wins - final_standings.wins
final_standings["gb_bye"] = final_standings["gb_bye"].astype(int)

fifth_seed_wins = final_standings.iloc[4,:].wins
final_standings["gb5"] = fifth_seed_wins - final_standings.wins
final_standings["gb5"] = final_standings["gb5"].astype(int)

sixth_seed_pf = final_standings.iloc[5,:].total_pf
final_standings["pb6"] = round(sixth_seed_pf - final_standings.total_pf, 2)

final_standings.iloc[:,-3:] = np.where(final_standings.iloc[:,-3:] < 0,
                    "+" + final_standings.iloc[:,-3:].mul(-1).astype(str),
                    final_standings.iloc[:,-3:].astype(str))

final_standings.iloc[:,-3:] = np.where((final_standings.iloc[:,-3:] == '0') | (final_standings.iloc[:,-3:] == '0.0'),
                    "–",
                    final_standings.iloc[:,-3:])

final_standings["total_pf"] = final_standings.total_pf.astype(float).map('{:.2f}'.format)
final_standings = final_standings[["team", "record", "win_perc", "total_pf", "gb_bye", "gb5", "pb6"]]
final_standings = final_standings.rename({"team":"Team",
                                          "record":"Record",
                                          "win_perc":"Win%",
                                          "total_pf":"Points",
                                          "gb_bye":"GB-B",
                                          "gb5":"GB-5",
                                          "pb6":"PB-6"}, axis=1)

headings_st = tuple(['Team', 'Record', 'Win%', 'Points', 'GB-Bye', 'GB-5', 'PB-6'])
data_st = [tuple(x) for x in final_standings.to_numpy()]

##### Power Ranks #####
pr_tables = power_table(league_id, season, swid, espn)
pr  = pr_tables[0]
pr = pr[pr.season == season]
pr["team"] = pr.team.str[:-1].str[:4]

# get table and plot
pr_table = pr_tables[1]
pr_table["team"] = pr_table.team.str[:-1].str[:4]
headings_pr = tuple(["Team", "Power Rank", "1 Week Change", "Power Score", "1 Week Change"])
data_pr = [tuple(x) for x in pr_table.to_numpy()]

# plot rank chart
rank_data = pr[["team", "week", "power_rank", "power_score"]].to_dict(orient="records")
rank_data = json.dumps(rank_data, indent=2)
rank_data = {"rank_data": rank_data}


##### Scenarios #####
# record vs each team and league
scen_tables = scenarios(d, pr[pr.season == season])
wins_vs_lg = scen_tables[0]
wins_vs_lg = get_wins_vs_lg(d, wins_vs_lg, pr, season).rename(columns={"team":"Team"})

headings_scen = tuple(wins_vs_lg.columns)
data_scen = [tuple(x) for x in wins_vs_lg.to_numpy()]

# record if you had another team's schedule
sched_switch = scen_tables[1]
sched_switch = get_sched_switch(d, sched_switch).rename(columns={"team":"Team"})

headings2_scen = tuple(sched_switch.columns)
data2_scen = [tuple(x) for x in sched_switch.to_numpy()]


##### Team Efficiency #####
pos_list = get_params(d)["lineup_slots_df"].pos.tolist()
pos_list = [x.lower() for x in pos_list]
pos_list = [x for x in pos_list if x not in ["bench", "ir"]]

# starting lineup efficiency
opt = get_optimal(d, league_id, season, swid, espn)

team_eff = team_efficiency(d, get_eff(opt, pos_list))
team_eff['team'] = team_eff['team'].str[:-1].str[:4]
team_eff['label'] = team_eff['team'] + ' (' + team_eff['effic'].astype(str) + ')'
adjust_text_dict = {'expand_points': (0.5, 2.0)}

eff_plot = (ggplot(aes(x="diffPerWeek", y="optPerWeek"), data=team_eff)
            + geom_point(aes(color="team"), size=4)
            #+ scale_y_continuous(breaks = range(min(eff_plot.optPerWeek), max(eff_plot.optPerWeek), 5))
            + geom_text(aes(label="label"), adjust_text=adjust_text_dict)
            + scale_color_discrete(guide=False)
            + annotate("segment",
                       x=min(team_eff.diffPerWeek)*1.02, xend=min(team_eff.diffPerWeek)*1.02,
                       y=min(team_eff.optPerWeek)*1.05, yend=max(team_eff.optPerWeek)*0.95,
                       arrow = arrow(type = "closed"), color="#999999")
            + annotate("segment",
                       x=min(team_eff.diffPerWeek)*0.9, xend=max(team_eff.diffPerWeek)*1.1,
                       y=max(team_eff.optPerWeek)*1.02, yend=max(team_eff.optPerWeek)*1.02,
                       arrow = arrow(type = "closed"), color="grey")
            + annotate("text",
                       x = min(team_eff.diffPerWeek)*1.03,
                       y = team_eff.optPerWeek.median(),
                       label = "Better Roster",
                       angle = 90, color="grey")
            + annotate("text",
                       x = team_eff.diffPerWeek.median()*1.02,
                       y = max(team_eff.optPerWeek)*1.03,
                       label = "Better Starts",
                       color="grey")
            + labs(x="Difference from Optimal Points per Week",
                   y="Optimal Points per Week")
            + theme_bw()
            + theme(axis_text = element_text(size=10),
                    axis_title = element_text(size=14)))

buf = io.BytesIO()
eff_plot.save(buf, format='png', width=8, height=6, verbose=False)
buf.seek(0)
buffer = b''.join(buf)
b2 = base64.b64encode(buffer)
eff_plot2 = b2.decode('utf-8')

# efficiency by position
pos_eff = pos_efficiency(get_eff(opt, pos_list))
pos_eff['team'] = pos_eff['team'].str[:-1].str[:4]
markers = [".", "s", "x", "*", "P", "d", "^"]
avg = pos_eff[pos_eff.Position == "TEAM"].avg_eff.mean()

data = pos_eff[pos_eff.variable != "team.eff"]

pos_plot = (ggplot(aes(x="avg_eff", y="team"), data=data)
            + geom_point(aes(color="Position", shape="Position"), size=3)
            + geom_vline(xintercept=avg, color="#999999", linetype="dashed", size=0.75)
            + scale_shape_manual(name="Position", values=markers)
            + scale_x_continuous(labels=lambda l: ["%d%%" % (v * 100) for v in l])
            + scale_color_brewer(type="qual", palette=2)
            + labs(x="Efficiency",
                   y="Team")
            + theme_bw()
            + theme(axis_text = element_text(size=10),
                    axis_title = element_text(size=14)))

buf2 = io.BytesIO()
pos_plot.save(buf2, format='png', width=8, height=6, verbose=False)
buf2.seek(0)
buffer2 = b''.join(buf2)
b3 = base64.b64encode(buffer2)
pos_plot2 = b3.decode('utf-8')
