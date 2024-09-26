import pandas as pd
from scipy.io import arff
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# EXERCISE 2
def arffToDf(file: str):
    data, meta = arff.loadarff(file) # meta contem o nome dos atributos que servem de colunas
    df = pd.DataFrame(data, columns=meta.names()) # converto num data frame com as colunas vindas do meta
    return df

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

    # PLOTTING
    # Plot class-conditional density plots for the best and worst features
    best_feature = anova_results.iloc[0]['Feature']
    worst_feature = anova_results.iloc[-1]['Feature']

    plt.figure(figsize=(14, 6))

    # Best feature plot
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=diabetes_df, x=best_feature, hue='Outcome', common_norm=False)
    plt.xlim(left=0)  # Set the x-axis lower limit to 0
    plt.title(f'Class-Conditional Density of Best Feature: {best_feature}')
    
    # Worst feature plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=diabetes_df, x=worst_feature, hue='Outcome', common_norm=False)
    plt.xlim(left=0)  # Set the x-axis lower limit to 0
    plt.title(f'Class-Conditional Density of Worst Feature: {worst_feature}')

    plt.tight_layout()
    plt.show()
    return anova_results

diabetes_df = arffToDf('diabetes.arff')
disc_rank_results = rankDiscriminativePower(diabetes_df)


# EXERCISE 2-4: Decision Tree classifier with varying min_samples_split

def decision_tree_analysis(df):

    # Ensure the Outcome column is categorical, else we get value error from cls calling
    df['Outcome'] = df['Outcome'].astype(str)

    # Split features and labels
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Stratified 80-20 train-test split with a fixed seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # Values for min_samples_split
    min_samples_splits = [2, 5, 10, 20, 30, 50, 100]

    # Lists to store accuracies
    train_accuracies = []
    test_accuracies = []

    # Train and evaluate Decision Tree for each min_samples_split
    for split in min_samples_splits:
        clf = DecisionTreeClassifier(min_samples_split=split, random_state=1)
        clf.fit(X_train, y_train)

        # Accuracy on training data appended to list
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        train_accuracies.append(train_acc)

        # Accuracy on testing data appended to list
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        test_accuracies.append(test_acc)

    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(min_samples_splits, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(min_samples_splits, test_accuracies, label='Testing Accuracy', marker='o')
    plt.xlabel('min_samples_split')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy vs. min_samples_split')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Exercise 4 - Train Decision Tree with max_depth=3 and random_state=1
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=1)
    clf.fit(X, y)

    plt.figure(figsize=(16, 10))
    plot_tree(clf, feature_names=X.columns, class_names=['Normal', 'Diabetes'], filled=True, rounded=True)
    plt.title('Decision Tree (max_depth=3)')
    plt.show()

# EXERCISE 2-4: Run Decision Tree analysis with varying min_samples_split
decision_tree_analysis(diabetes_df)