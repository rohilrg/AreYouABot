import numpy as np
import pandas as pd

from src.evaluation_functions import evaluation_metrics
from sklearn.model_selection import train_test_split
from src.feature_engineering import TransformationOfLogWithNewFeatures
from sklearn.tree import DecisionTreeClassifier
import joblib

print('Welcome to the AreYouABot program:')
to_continue = input("Enter 1 to continue or 0 to exit: ")
if to_continue == '1':
    print('Now we will create a train and test set with files in the folder: data/raw_dataset')

    print('Transforming  overall train files..............')

    trainset_overall\
        ,total_distinct_categories_in_log\
        , average_events_done_by_typical_real_user\
        = TransformationOfLogWithNewFeatures('data/raw_dataset/train/fake_users.csv').run_transformer_for_train()

    print('Saving the overall train file in folder data/overall_train_file')

    trainset_overall.to_csv('data/overall_train_file/overall_train_file.csv', index=False)
    print('__________________________________________________')
    print('Transforming done, now splitting the data into train and validation set')

    X_train, X_val, y_train, y_val = train_test_split(
        trainset_overall.iloc[:, 1:-1], trainset_overall.iloc[:, -1], test_size=0.10, random_state=42)

    print('Splitting done, now saving files')
    np.save('data/splitted_dataset_for_training_and_validation/train/features.npy', X_train)
    np.save('data/splitted_dataset_for_training_and_validation/val/features.npy', X_val)

    np.save('data/splitted_dataset_for_training_and_validation/train/labels.npy', y_train)
    np.save('data/splitted_dataset_for_training_and_validation/val/labels.npy', y_val)

    print('__________________________________________________')

    print('Now we will transform the test set also')

    test_set_transformed =\
        TransformationOfLogWithNewFeatures('data/raw_dataset/test/fake_users_test.csv').run_transformer_for_test(
            total_distinct_categories_in_log=total_distinct_categories_in_log,
            average_events_done_by_typical_real_user=average_events_done_by_typical_real_user
        )

    print('Transformation done, now saving the test set in folder: data/test')

    test_set_transformed.to_csv('data/test/test_set_transformed.csv', index=False)

    np.save('data/test/features.npy', test_set_transformed.iloc[:, 1:-1])
    np.save('data/test/labels.npy', test_set_transformed.iloc[:,-1])
    print('__________________________________________________')

    train_a_model = input('Do you want to train the model? 0 or 1: ')

    if train_a_model == '1':
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)

        joblib.dump(clf, 'trained_model/model.pkl')

        print('The evaluation metrics on validation set are:')
        y_pred_val = clf.predict(X_val)

        evaluation_metrics(y_val, y_pred_val, suffix_for_file='validation')
        print('__________________________________________________')

        print('The evaluation metrics on test set are:')
        y_pred_test = clf.predict(test_set_transformed.iloc[:, 1:-1])
        is_fake_probablity = clf.predict_proba(test_set_transformed.iloc[:, 1:-1])[:,1]
        output_df = pd.DataFrame(data=
                                 {'UserId': test_set_transformed.iloc[:, 0],
                                  'is_fake_probablity': is_fake_probablity})

        output_df.to_csv('output/test_set_predictions.csv', index=False)
        evaluation_metrics(test_set_transformed.iloc[:,-1], y_pred_test, suffix_for_file='test')
    else:
        print('Do want to see the evaluation metrics the pretrained model?')
        yes_or_no = input('Enter 1 for yes and 0 for no.: ')
        if yes_or_no == '1':
            clf = joblib.load('trained_model/model.pkl')

            print('The evaluation metrics on validation set are:')
            y_pred_val = clf.predict(X_val)

            evaluation_metrics(y_val, y_pred_val, suffix_for_file='validation')
            print('__________________________________________________')

            print('The evaluation metrics on test set are:')
            y_pred_test = clf.predict(test_set_transformed.iloc[:, 1:-1])
            is_fake_probablity = clf.predict_proba(test_set_transformed.iloc[:, 1:-1])[:,1]

            output_df = pd.DataFrame(data=
                                     {'UserId': test_set_transformed.iloc[:, 0],
                                      'is_fake_probablity': is_fake_probablity})

            output_df.to_csv('output/test_set_predictions.csv', index=False)

            evaluation_metrics(test_set_transformed.iloc[:, -1], y_pred_test, suffix_for_file='test')

        else:
            print('Bye!! Have a nice day.')
else:
    print('Bye!! Have a nice day.')




