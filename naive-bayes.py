import pandas as pd
import numpy as np


def naive_bayes(data):

    # Split DataFrame by class to prepare for stratified cross validation
    negative_df, positive_df = [x for _, x in data.groupby(data.Class == 'positive')]
    # Split data for each class into 5 folds
    negative_folds = np.split(negative_df[:-(negative_df.shape[0] % 5)], 5)
    positive_folds = np.split(positive_df[:-(positive_df.shape[0] % 5)], 5)
    # Combine positive and negative folds
    # We will then have even distribution of classes for stratified 5-fold CV
    folds = []
    for i in range(len(negative_folds)):
        folds.append(pd.concat(list([negative_folds[i], positive_folds[i]])))
        # Sample all data within fold to shuffle
        folds[i] = folds[i].sample(frac=1)

    # Iterate through folds for cross validation
    for i in range(len(folds)):

        # Copy original data to work with in the loop
        working_data = folds.copy()
        # Create training and testing sets
        test = working_data.pop(i)
        test = test.drop('Class', axis=1)
        # print(test.shape)
        train = pd.concat(working_data)
        # print(train.shape)

        # Find class probabilities
        class_counts = train.Class.value_counts()
        # Create dict to track class values and their count
        class_counts = dict(zip(class_counts.index, class_counts))
        total_count = sum(class_counts.values())
        # Calculation of class priors
        prior_negative = class_counts['negative'] / total_count
        prior_positive = class_counts['positive'] / total_count
        # print(prior_positive, " ", prior_negative, " ", (prior_negative + prior_positive))

        # Iterate through features to calculate probabilities with all their values
        for feature, values in train.iloc[:, :-1].iteritems():
            value_counts = values.value_counts()
            # Create dict to track feature values and their counts
            value_counts = dict(zip(value_counts.index, value_counts))

            print(value_counts)

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

# Rename class column to 'class'
dataset.rename(columns={"Subscribed": "Class"}, inplace=True)

# Convert classes to 'positive' and 'negative'
dataset.Class.replace(to_replace=['yes', 'no'], value=['positive', 'negative'], inplace=True)

naive_bayes(dataset)
