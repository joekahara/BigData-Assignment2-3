import pandas as pd
import numpy as np

# pd.set_option('display.max_rows', 1000)


def naive_bayes(df):
    """class_count = df.iloc[:,-1].count()
    prior_positive = df.iloc[:,-1][df.iloc[:,-1] == 'positive'].count() / class_count
    prior_negative = df.iloc[:,-1][df.iloc[:,-1] == 'negative'].count() / class_count
    # print(prior_positive, " ", prior_negative, " ", (prior_negative + prior_positive))"""

    prior_positive = 1/2.0
    prior_negative = 1/2.0


df = pd.read_csv("naive-bayes-data.csv")

# Encode numeric age to a categorical feature (grouping not perfectly accurate, estimated)
categorical_ages = pd.cut(df.Age,
                          bins=[17,25,35,40,50,65,99],
                          labels=['17-25', '25-35', '35-40', '40-50', '50-65', '65-99'])
df.insert(0, "Age_Group", categorical_ages)
df.drop(columns=['Age'], inplace=True)
print(df.Age_Group.value_counts().sort_index())

print(df)

# Convert classes to 'positive' and 'negative'
df.Subscribed.replace(to_replace=['yes', 'no'], value=['positive', 'negative'], inplace=True)

naive_bayes(df)
