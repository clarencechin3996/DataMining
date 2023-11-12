import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier



def train_decision_tree(X_train, y_train, X_test, y_test, best_params):
    # Initialize Decision Tree Classifier with best parameters
    best_dt_classifier = DecisionTreeClassifier(**best_params, random_state=30)

    # Train the model
    best_dt_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = best_dt_classifier.predict(X_test)

    # Evaluate the model and generate metrics
    test_score = best_dt_classifier.score(X_test, y_test)
    f1_test_score = f1_score(y_test, y_pred, average='macro')

    return best_dt_classifier, test_score, f1_test_score



def train_random_forest(X_train, y_train, X_test, y_test, best_params_rf):
    # Initialize Random Forest Classifier with best parameters
    best_rf_classifier = RandomForestClassifier(**best_params_rf, random_state=50)

    # Train the model
    best_rf_classifier.fit(X_train, y_train)

    # Predictions
    y_pred_rf = best_rf_classifier.predict(X_test)

    # Evaluate the model and generate metrics
    test_score_rf = best_rf_classifier.score(X_test, y_test)
    f1_test_score_rf = f1_score(y_test, y_pred_rf, average='macro')

    return best_rf_classifier, test_score_rf, f1_test_score_rf



def train_knn(X_train, y_train, X_test, y_test, best_params_knn):

    # Initialize k-NN Classifier with best parameters
    best_knn_classifier = KNeighborsClassifier(**best_params_knn)

    # Train the model
    best_knn_classifier.fit(X_train, y_train)

    # Predictions
    y_pred_knn = best_knn_classifier.predict(X_test)

    # Evaluate the model and generate metrics
    test_score_knn = best_knn_classifier.score(X_test, y_test)
    f1_test_score_knn = f1_score(y_test, y_pred_knn, average='macro')

    return best_knn_classifier, test_score_knn, f1_test_score_knn




def train_naive_bayes(X_train, y_train, X_test, y_test, best_params_NB):
    # Initialize Naive Bayes Classifier with best parameters
    naive_bayes_classifier = GaussianNB(var_smoothing=best_params_NB['var_smoothing'])

    # Train the model
    naive_bayes_classifier.fit(X_train, y_train)

    # Predictions
    y_pred_nb = naive_bayes_classifier.predict(X_test)

    # Evaluate the model and generate metrics
    test_score_nb = naive_bayes_classifier.score(X_test, y_test)
    f1_test_score_nb = f1_score(y_test, y_pred_nb, average='macro')

    return naive_bayes_classifier, test_score_nb, f1_test_score_nb



def generate_predictions_and_csv(model, X_test, filename, accuracy, f1_score, verbose=False):
    y_pred = model.predict(X_test)[:300]
    result_df = pd.DataFrame({'Predictions': y_pred})

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({'Accuracy': [round(accuracy, 3)], 'Macro-F1': [round(f1_score, 3)]})

    # Concatenate the predictions and metrics DataFrames
    final_df = pd.concat([result_df, metrics_df], ignore_index=True)

    # Save to CSV
    final_df.to_csv(filename, index=False)

    if verbose:
        print(f"Saved predictions and metrics to {filename}")


# Function to calculate average F1-score using cross-validation
def get_avg_f1_score(model, X, y, cv=5):
    stratified_k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=312)
    f1_scores = cross_val_score(model, X, y, cv=stratified_k_fold, scoring='f1_macro')
    return f1_scores.mean()




def main():
    # load the dataset
    train_data = pd.read_csv('/Users/clarencechin/Desktop/DS/INFS4203/train.csv')
    test_data = pd.read_csv('/Users/clarencechin/Desktop/DS/INFS4203/test.csv')

    # Pre-processing code
    # Identify numerical and categorical columns
    numerical_columns = train_data.columns[:100].tolist()
    categorical_columns = train_data.columns[100:-1].tolist()

    # Train-Test Split
    X = train_data.drop(columns=['Label'])
    y = train_data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize imputers
    mean_imputer = SimpleImputer(strategy='mean')
    mode_imputer = SimpleImputer(strategy='most_frequent')

    # Fit the imputers on the training data
    mean_imputer.fit(X_train[numerical_columns])
    mode_imputer.fit(X_train[categorical_columns])

    # Apply Imputation on Training set
    X_train_imputed = X_train.copy()
    X_train_imputed[numerical_columns] = mean_imputer.transform(X_train[numerical_columns])
    X_train_imputed[categorical_columns] = mode_imputer.transform(X_train[categorical_columns])

    # Apply Imputation on Test set
    X_test_imputed = X_test.copy()
    X_test_imputed[numerical_columns] = mean_imputer.transform(X_test[numerical_columns])
    X_test_imputed[categorical_columns] = mode_imputer.transform(X_test[categorical_columns])

    # Show some sample rows from imputed datasets to confirm
    X_train_imputed.head(), X_test_imputed.head()

    # Initialize the MinMax Scaler
    minmaxscaler = MinMaxScaler()

    # Step 3:
    # Normalize the numerical features on both training and test sets using MinMaxScaler
    X_train_normalized_minmax = X_train_imputed.copy()
    X_train_normalized_minmax[numerical_columns] = minmaxscaler.fit_transform(X_train_imputed[numerical_columns])

    X_test_normalized_minmax = X_test_imputed.copy()
    X_test_normalized_minmax[numerical_columns] = minmaxscaler.transform(X_test_imputed[numerical_columns])

    # Initialize the Standard Scaler and Isolation Forest for outlier detection
    scaler = StandardScaler()
    iso_forest = IsolationForest(contamination=0.1, random_state=42)

    # Normalize the numerical features on the imputed Training set
    X_train_normalized = X_train_imputed.copy()
    X_train_normalized[numerical_columns] = scaler.fit_transform(X_train_imputed[numerical_columns])

    # Normalize the numerical features on the imputed Test set
    X_test_normalized = X_test_imputed.copy()
    X_test_normalized[numerical_columns] = scaler.transform(X_test_imputed[numerical_columns])

    # Step 4: Apply outlier detection only on training set to avoid data leakage
    outliers = iso_forest.fit_predict(X_train_normalized[numerical_columns])
    X_train_no_outliers = X_train_normalized[outliers == 1]
    y_train_no_outliers = y_train[outliers == 1]

    # Show the number of outliers detected and the shape of the new dataset
    num_outliers = len(X_train_normalized) - len(X_train_no_outliers)

    # Show number of outliers and result of after-processed
    # print("Number of outliers: ",num_outliers)
    # print("X_train_no_outliers.shape: ",X_train_no_outliers.shape)




    # Best hyperparameters directly (You would replace this with your actual best params)
    best_params_df = {
        'criterion': 'gini',
        'max_depth': 9,
        'min_samples_leaf': 4,
        'min_samples_split': 5,
        'splitter': 'best'
    }

    # Train the model and get the metrics for Decision Tree without normalization
    best_dt_classifier, test_score_dt, f1_test_score_dt = train_decision_tree(X_train_no_outliers, y_train_no_outliers, X_test_normalized, y_test, best_params_df)
    avg_f1_dt = get_avg_f1_score(best_dt_classifier, X_train_no_outliers, y_train_no_outliers)
    print("\nDecision Tree - Accuracy :", test_score_dt)
    print("Decision Tree - Average F1 Score on Test Set :", avg_f1_dt)




    # Best hyperparameters for Random Forest
    best_params_rf = {
        'n_estimators': 50,
        'max_depth': 7,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'bootstrap': True,
        'max_features': None
    }

    # Train the model and get the metrics for Random Forest without normalization
    best_rf_classifier, test_score_rf, f1_test_score_rf = train_random_forest(X_train_no_outliers, y_train_no_outliers, X_test_normalized, y_test, best_params_rf)
    avg_f1_rf = get_avg_f1_score(best_rf_classifier, X_train_no_outliers, y_train_no_outliers)

    print("\nRandom Forest - Accuracy :", test_score_rf)
    print("Random Forest - Average F1 Score on Test Set :", avg_f1_rf)







    # Best hyperparameters for k-NN (replace this with your actual best params)
    best_params_knn = {
        'algorithm': 'auto',
        'leaf_size': 20,
        'metric': 'euclidean',
        'n_neighbors': 65,
        'weights': 'uniform'
    }

    # Train the model and get the metrics for k-NN with normalized data
    best_knn_classifier, test_score_knn, f1_test_score_knn = train_knn(X_train_no_outliers, y_train_no_outliers, X_test_normalized, y_test, best_params_knn)
    avg_f1_knn = get_avg_f1_score(best_knn_classifier, X_train_no_outliers, y_train_no_outliers)
    print("\nk-NN - Accuracy:", test_score_knn)
    print("k-NN - Average F1 Score on Test Set:", avg_f1_knn)







    # Best hyperparameters for Naive Bayes
    best_params_NB = {
        'var_smoothing': 0.0657933224657568
    }

    # Train the model and get the metrics for Naive Bayes
    best_NB, test_score_NB, f1_test_score_NB = train_naive_bayes(
        X_train_imputed, y_train, X_test_imputed, y_test, best_params_NB
    )
    avg_f1_nb = get_avg_f1_score(best_NB, X_train_no_outliers, y_train_no_outliers)

    print("\nNaive Bayes - Accuracy:", test_score_NB)
    print("Naive Bayes - Average F1 Score on Test Set:", avg_f1_nb)






    test_data_imputed = test_data.copy()

    # Normalize the numerical features on the imputed Test set
    test_data_normalized = test_data_imputed.copy()
    test_data_normalized[numerical_columns] = scaler.transform(test_data_imputed[numerical_columns])

    # Create a voting classifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('dt', best_dt_classifier),
            ('rf', best_rf_classifier),
            ('knn', best_knn_classifier),
            ('nb', best_NB)
        ],
        voting='hard'
    )

    # Train ensemble model on the training set
    ensemble_model.fit(X_train_no_outliers, y_train_no_outliers)

    # Evaluate on the test set
    ensemble_test_score = ensemble_model.score(X_test_normalized, y_test)
    avg_f1_ensemble = get_avg_f1_score(ensemble_model, X_train_no_outliers, y_train_no_outliers)
    print("--------------------------------------------------------------------------------------------------------")
    print(f'Ensemble Accuracy Score: {ensemble_test_score}')
    print(f'Ensemble Average F1 Score: {avg_f1_ensemble}')

    # Generate and save ensemble predictions
    generate_predictions_and_csv(ensemble_model, test_data_normalized, 's4761962_ensemble.csv', ensemble_test_score, avg_f1_ensemble, verbose=True)

    # Generate predictions for each model and save to CSV
    generate_predictions_and_csv(best_dt_classifier, test_data_normalized, 's4761962_dt.csv', test_score_dt, avg_f1_dt)
    generate_predictions_and_csv(best_rf_classifier, test_data_normalized, 's4761962_rf.csv', test_score_rf, avg_f1_rf)
    generate_predictions_and_csv(best_knn_classifier, test_data_normalized, 's4761962_knn.csv', test_score_knn, avg_f1_knn)
    generate_predictions_and_csv(best_NB, test_data_imputed, 's4761962_nb.csv', test_score_NB, avg_f1_nb)

    # Generate a CSV file for the results (You would include this step after generating y_pred)
    # result_df = pd.DataFrame({'Predictions': y_pred})
    # result_df.to_csv('sxxxxxxx.csv', index=False)



if __name__ == '__main__':
    main()


