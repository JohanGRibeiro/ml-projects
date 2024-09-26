from sklearn.feature_selection import f_classif
from scipy.io import arff
import pandas as pd
import os
print(os.getcwd())

import matplotlib.pyplot as plt
import seaborn as sns

# importing data

def arffToDf(file: str):
    data, meta = arff.loadarff(file) # meta contem o nome dos atributos que servem de colunas
    df = pd.DataFrame(data, columns=meta.names()) # converto num data frame com as colunas vindas do meta
    print(meta.names())
    return df


# 1) Using f_classif from sklearn, identify the input variables with the
# worst and best discriminative power

def rankDiscriminativePower(df):
    # Split features and labels
    X = df.drop(columns=['Outcome'])  # (drop method removes the specified column)
    y = df['Outcome'] 

    F_values, p_values = f_classif(X, y)
    # Sort results with dataframe
    anova_results = pd.DataFrame({'Feature': X.columns, 'F_value': F_values, 'p_value': p_values})
    anova_results = anova_results.sort_values(by='F_value', ascending=False)

    # Display the features with the best and worst discriminative power
    print("Feature ranking for discrimnative power: \n", anova_results.head(10))
    print("Best feature:\n", anova_results.head(1))
    print("Worst feature:\n", anova_results.tail(1))

    return anova_results

# Plot class-conditional density plots for the best and worst features

def plotDensity(df, anova_results):
    # PLOTTING
    # Plot class-conditional density plots for the best and worst features
    best_feature = anova_results.iloc[0]['Feature']
    worst_feature = anova_results.iloc[-1]['Feature']

    plt.figure(figsize=(14, 6))
    # Best feature plot
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x=best_feature, hue='Outcome', common_norm=False)
    plt.title(f'Class-Conditional Density of Best Feature: {best_feature}')
    # Worst feature plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df, x=worst_feature, hue='Outcome', common_norm=False)
    plt.title(f'Class-Conditional Density of Worst Feature: {worst_feature}')

    plt.tight_layout()
    plt.show()
    return

df = arffToDf('diabetes.arff')
disc_rank_results = rankDiscriminativePower(df)
plotDensity(df, disc_rank_results)






# 2) Using a stratified 80-20 training-testing split with a fixed seed (random_state=1), assess
#in a single plot both the training and testing accuracies of a decision tree with minimum
#sample split in {2, 5,10, 20, 30, 50, 100} and the remaining parameters as default.
#Note that split thresholding of numeric variables in decision trees is nondeterministic in sklearn, hence you may opt to average the results using 10 runs per
#parameterization.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder


def trainTestSplit(df, min_samples_split, n_runs=10):
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Encode the target variable if necessary
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    train_accuracies = []
    test_accuracies = []

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Define model
        model = DecisionTreeClassifier(min_samples_split=min_samples_split)
        model.fit(X_train, y_train)

        # Define prediction arrays
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Get accuracy score from prediction arrays
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Append them to the lists
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)

    return avg_train_accuracy, avg_test_accuracy

def plotTrainTestAccuracies(df):
    min_samples_splits = [2, 5, 10, 20, 30, 50, 100]

    train_accuracies = []
    test_accuracies = []

    for min_samples_split in min_samples_splits:
        train_acc, test_acc = trainTestSplit(df, min_samples_split)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    plt.plot(min_samples_splits, train_accuracies, label='Training Accuracy')
    plt.plot(min_samples_splits, test_accuracies, label='Testing Accuracy')
    plt.xlabel('Minimum Samples Split')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracies of Decision Tree')
    plt.legend()
    plt.show()

plotTrainTestAccuracies(df)