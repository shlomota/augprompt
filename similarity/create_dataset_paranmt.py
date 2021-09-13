import pandas as pd

from sklearn.model_selection import train_test_split
sample_size = 20000
# df = pd.read_csv(r"C:\Users\soki\PycharmProjects\augprompt\data\para-nmt-5m-processed\para-nmt-5m-processed.txt", sep="\t", index_col=False, names=["text1", "text2"])
df = pd.read_csv("../data/para-nmt-5m-processed/para-nmt-5m-processed.txt", sep="\t", index_col=False, names=["text1", "text2"])
df_both = df.sample(sample_size)
df_1, df_2 = train_test_split(df_both, test_size=0.5)
df_1["label"] = [1] * len(df_2)
df_2 = df_2.reset_index(drop=True)
df_2["text2"] = df_2["text2"].sample(frac=1).reset_index(drop=True)
df_2["label"] = [0] * len(df_2)
df_merged = df_1.append(df_2).sample(frac=1).reset_index(drop=True)
df_merged.to_csv("./data/para-nmt-5m-processed/para-nmt-balanced-%d.txt" % (sample_size), sep="\t", index=False)
