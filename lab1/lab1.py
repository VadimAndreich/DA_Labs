import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()

df = pd.read_csv("dataset/iris.csv")
names = df.columns.tolist()

df[names[-1]] = encoder.fit_transform(df[names[-1]])
correlation = df.corr()

grouped_df = df.groupby(names[-1])
correlation_grouped = grouped_df.corr()

sns.pairplot(df)
plt.show()
