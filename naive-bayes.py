import pandas as pd
import numpy as np


def naive_bayes(data):
    # Split dataset into 5 folds for 5-fold cross validation
    # Drop n rows during split to make the length of data divisible by 5
    data = np.split(data[:-(data.shape[0] % 5)], 5)

    # Iterate through folds for cross validation
    for fold in data:
        # Create test and training sets
        test = fold.drop('Subscribed', axis=1)
        print(test.shape)
        train = pd.concat(data.all(not fold))
        print(train.shape)
        # Find class probabilities
        class_counts = np.unique(train[:, -1], return_counts=True)
        total_count = sum(class_counts[1])
        prior_negative = class_counts[1][0] / total_count
        prior_positive = class_counts[1][1] / total_count
        print(prior_positive, " ", prior_negative, " ", (prior_negative + prior_positive))


# Read phone marketing data from CSV file
dataset = pd.read_csv("naive-bayes-data.csv")

# Encode numeric age to a categorical feature (grouping not perfectly accurate, estimated)
categorical_ages = pd.cut(dataset.Age,
                          bins=[17, 25, 35, 40, 50, 65, 99],
                          labels=['17-25', '25-35', '35-40', '40-50', '50-65', '65-99'])
dataset.insert(0, "Age_Group", categorical_ages)
dataset.drop(columns=['Age'], inplace=True)

# print("Distribution among age categories:")
# print(df.Age_Group.value_counts().sort_index(), "\n")

# Convert classes to 'positive' and 'negative'
dataset.Subscribed.replace(to_replace=['yes', 'no'], value=['positive', 'negative'], inplace=True)

naive_bayes(dataset)
