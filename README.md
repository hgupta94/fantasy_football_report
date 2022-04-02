# [Link to report](https://www.thechillff.com/)

League report for my personal Fantasy Football league, created using Python and Flask. Includes a power ranking, simulation results, what-if record scenarios, and team efficiency charts

# Power Ranking
A combination of five factors are used to calcuate a weekly Power Score for each team, which is then normalized so that 100 is an average team. 
1. **Wins**: Total wins on the season.
2. **Season Performance**: Total points scored on the season.
3. **Recent Performance**: Total points scored over the past three weeks.
4. **Consistency**: How consistent each team is at scoring, weighted by average points per week so consistent low scoring teams are penalized more than consistent high scoring teams.
5. **Luck**: Actual weekly result (win/loss) compared to how many teams you outscored each week. 

# Simulations
A monte carlo simulation is ran at the individual player level to simulate current week matchups, final standings, and playoff outcomes.
## Current Week
Current week matchups are simulated 1000 times. The following steps are taken to arrive at the final outcome:
1. Each team's optimal lineup is selected based on aggregate projections from FantasyPros. The flex position (running back/wide receiver/tight end) is a random selection between the third highest projected running back and wide receiver, since tight ends are rarely selected, to add additional variance.
2. Player scores are then drawn from a normal distribution, using the FantasyPros projections as the mean and a set percentage of that mean as the standard deviation. Since quarterbacks are more consistent, their standard deviation is 20% of their projection while all other players are 40%.
3. To account for the inherent randomness in fantasy football, a random number of points between -10 and 30 are added to each player.

These two scores are then weighted 70% and 30% respectively, summed to arrive at an overall team score, and matchup outcomes are then determined. After 1000 simulations are ran, a summarized betting table is created displaying likelihood and odds of each team winning, average score, and odds of each team being the highest and lowest scorer.

## Regular Season and Playoffs
The regular season and playoffs are also simulated 1000 times.
1. Optimal lineups are selected using rest-of-season projections from numberFire. Similar to the current week simulation, the flex position is randomly chosen between the third highest projected RB and WR. Bye weeks and injuries are taken into account. All players projected to score under 4 points are removed, since these players are generally no better than free agents. If a particular lineup slot cannot be filled, the 75th percentile player for that position is selected to represent a typical free agent player.
2. Each team's average points per week and standard deviation are used to draw a score from a normal distribution.
3. Using the same method as #2 in the Current Week simulation, player scores are simulated using a normal distribution.
4. A random number of points between 70 and 120, roughly the low and high scores in a given week, are added.

Prior to week 4, #3 and #4 are weighted 60% and 40%, respectively; from week 4 on, the weights change to 20%, 50%, and 30%. Using these scores, each remaining regular season matchup is simulated. The top four teams in each simulation move on to the playoffs. The 1/4 and 2/3 seeds face off for a two week semifinal and the winners move on to a two week finals. Each team's regular season final rank and win total are counted, along with the number of playoff and finals appearances and the number of first, second, and third place finishes. 

# What If Scenarios
## Record vs League
Rather than a matchup-based schedule, this scenario aims to see how each team would fare if they played every team each week. 

## Schedule Switcher
Unlike the NFL schedule, fantasy football schedules are randomly generated. As a result, luck plays a large part in every team's success (or failure). Score the second most points and lose? Tough luck. Score the second *least* points and win? Well, a win's a win! This table shows what each team's record would be if they instead had another team's schedule.

# Team Efficiencies
A perfect lineup is tough to acheive. This page displays each team's overall lineup efficiency as well as by position.
