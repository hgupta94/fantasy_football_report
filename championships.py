import pandas as pd

champs = pd.read_csv(r"C:\Users\hirsh\OneDrive\Desktop\Data Science Stuff\Projects\FF Leagues\FF Analysis\Flask_Cool\tables\champions.csv")
champs["Count"] = champs.groupby("Champion").cumcount()+1
