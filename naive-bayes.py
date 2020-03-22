import pandas as pd
import numpy as np


def naive_bayes(data):
    # Set m for m-estimate naive-bayes arbitrarily
    m = 30

    # ---------------- Cross Validation Prep ---------------- #
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
        train = pd.concat(working_data)

        # ---------------- Probability Calculations ---------------- #
        # Find distinct classes
        class_values = train.Class.unique()
        # Calculation of class priors
        prior_probabilities = train.Class.value_counts() / train.shape[0]

        conditional_probabilities = {}
        # Iterate through features to calculate probabilities with all their values
        for feature, values in train.iloc[:, :-1].iteritems():
            feature_values = values.unique()
            # Initialize numpy array with 0s to store feature/class value conditional probability
            feature_conditional_probability = dict.fromkeys(class_values, dict.fromkeys(feature_values))
            for j in range(len(class_values)):
                for k in range(len(feature_values)):
                    # Calculate conditional probability for each feature value with both classes
                    # P(x|c) = (n_c + mp) / (n + m)
                    # x = feature value, c = class, n_c = # of feature values for given class,
                    # p = class prior, n = # of class values for given class
                    feature_conditional_probability[class_values[j]][feature_values[k]] = \
                        ((train.loc[(values == feature_values[k]) &
                                    (data.Class == class_values[j]), ].shape[0]) / \
                        (sum(train.Class == class_values[j])))
            conditional_probabilities[feature] = feature_conditional_probability

        # ---------------- Prediction ---------------- #
        predictions = []
        # Iterate through test set rows to predict each one's class
        for row in test.itertuples(index=False):
            # Initialize product of feature values' conditional probabilities
            negative_conditionals, positive_conditionals = 1, 1
            for j in range(test.iloc[:, :-1].shape[1]):
                # For each feature, continue to calculate the product
                negative_conditionals *= conditional_probabilities[test.columns[j]][class_values[0]][row[j]]
                positive_conditionals *= conditional_probabilities[test.columns[j]][class_values[1]][row[j]]
            # Calculate prediction probabilities
            prob_of_negative = prior_probabilities[0] * negative_conditionals
            prob_of_positive = prior_probabilities[1] * positive_conditionals
            # Predict class according to calculated probabilities
            if prob_of_negative > prob_of_positive:
                predictions.append(class_values[0])
            else:
                predictions.append(class_values[1])
        # Add predictions to the test dataset then print
        test = test.assign(Prediction=predictions)
        print(test)

        # ---------------- Performance Metrics ---------------- #
        # Count and print confusion matrix
        tp, tn, fp, fn = 0, 0, 0, 0
        for j in range(test.shape[0]):
            if test.iloc[j, -1] == 'negative':
                if test.iloc[j, -2] == 'negative':
                    tn += 1
                else:
                    fn += 1
            else:
                if test.iloc[j, -2] == 'positive':
                    tp += 1
                else:
                    fp += 1
        print("TP: ", tp, " TN: ", tn, " FP: ", fp, " FN: ", fn)
        # Calculate and print performance metrics
        accuracy = (tp + tn) / test.shape[0]
        if tp == 0 or fp == 0:
            sensitivity = 'NaN'
        else:
            sensitivity = tp / (tp + fp)
        if tn == 0 or fp == 0:
            specificity = 'NaN'
        else:
            specificity = tn / (tn + fp)
        print("Accuracy = ", accuracy, ", sensitivity = ", sensitivity, ", specificity = ", specificity)
        # Combine test and training sets to recreate original dataset with added predictions
        pd.concat(list([test, train])).to_csv('predictions.csv')


# Read phone marketing data from CSV file
dataset = pd.read_csv("naive-bayes-data.csv")

# Encode numeric age to a categorical feature (grouping not perfectly accurate, estimated)
categorical_ages = pd.cut(dataset.Age,
                          bins=[16, 25, 35, 40, 50, 65, 99],
                          labels=['16-25', '25-35', '35-40', '40-50', '50-65', '65-99'])
dataset.insert(0, "Age_Group", categorical_ages)
dataset.drop(columns=['Age'], inplace=True)

# Rename class column to 'class'
dataset.rename(columns={"Subscribed": "Class"}, inplace=True)

# Convert classes to 'positive' and 'negative'
dataset.Class.replace(to_replace=['yes', 'no'], value=['positive', 'negative'], inplace=True)

naive_bayes(dataset)
