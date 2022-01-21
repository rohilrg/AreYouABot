# AreYouABot
A classifier to detect bots that produces fake clicks and leads on the website.

## Pre-installation steps:
- Make sure you have python 3+ installed on your computer.
- Make sure you have pip3 package installed on your computer.

## Install requirements: 
In your terminal/command-line go to the project folder and execute the command below:
```bash
pip3 install -r InstallMe.txt 
```

## Running of the project:
Please run the following command in your terminal and read carefully to proceed thereafter.
```bash
python main.py
```

## Internal working of the project:
a. This project takes the files stored in data/raw_dataset/ to transform them to create a train, val and test set.

b. There are 4 features that are constructed: 
1. No_of_events: It adds a feature to dataframe mentioning the number of events
        each user did on it's arrival on the website.
2. Ratio_of_categories_interacted: It adds the ratio of no. of distinct category user interacted upon total number
        of distinct category interacted by all the user.
3. Ratio_of_skewness_of_events_to_detect_bot_behaviour: It adds the ratios of click_ad/total_events, send_email/total_events and
        send_sms/total_events. These features will give the information about the typical bot
        behaviour.
4. Ratio_of_amount_events_done_more_by_a_typical_bot: It adds the ratio of
        (total events done by a user - average events done by real user)/(average events done by real user)
        This feature is designed to encapsulate the typical bot behaviour, as for a bot this value will tend
        to be more than 1 and for a real user it will lie between 0 and 1 in normal cases.

c. Thereafter, a simple DecisionTreeClassifier is trained with default parameters and the trained model is saved at trained_model/model.pkl.

d. The evaluation metrics are printed when main.py is running for validation and test set. 

e. The confusion matrix plots are saved in folder plots/

f. The output of the test set as required is saved in folder  output/test_set_predictions.csv. 

## Results on Validation and Test set:
### Validation-set
![Validation-set](https://github.com/rohilrg/AreYouABot/blob/main/plots/confusion_matrix_for_validation_set.png)
### Test-set
![Test-set](https://github.com/rohilrg/AreYouABot/blob/main/plots/confusion_matrix_for_test_set.png)
