import pandas as pd

df = pd.DataFrame(data=[["1.jpg",1.6],["2.jpg",0.3]], columns=["c1","c2"])
df.to_csv("data/csv.csv", index=False)