# Kaggle Titanic - Machine Learning from Disaster

## Overview
In this project, I have been given data of the passengers who were onboard the Titanic and am tasked to use machine learning techniques to predict whether passengers in the test dataset survivied the infamous Titanic shipwreck which took place on April 15, 1912.

I used random forest classification to predict the 'y' variable, 'Survived', using  dataset which contained the following 'x' variables:

|Feature      |                                            Data Dictionary Description                                                   |
|-------------|--------------------------------------------------------------------------------------------------------------------------|
|'Pclass'     |Ticket class. 1=1st, 2=2nd, 3=3rd                                                                                         |
|'Sex'        |Sex (of passenger)                                                                                                        |
|'Age'        |Age of passenger in years                                                                                                 |
|'Fare'       |Passenger fare                                                                                                            |
|'Embarked'   |Port of Embarkation                                                                                                       |
|'HasCabin'   |Indicates whether passenger's 'Cabin' value was null. 1 = not null, 0 = null                                              |
|'PartyBinned'|Size of passenger's party Alone=1 passenger, Couple=2 passenger, SmallMedFamily=3-5 passengers, LargeFamily= >5 passengers|
|'Title'      |Indicates passenger's title                                                                                               |

In the end, the model featured in this repository gave me a final accuracy score of **0.74401**. I have actually undertaken this project several times and am unsure of how I can raise my accuracy score higher than it currently is. Am open to feedback or suggestions on how to improve!

## Data Cleaning & Engineering
### Train Set
There were several null values that required addressing:

1. **'Embarked'**: I manually imputed 2 null values based on publicly available information on the internet
2. **'Age'**: I imputed the ~20% null values with the median age according to 'Pclass' and 'Sex'

77% of values under 'Cabin' were also null. Because no other features provided me with any clues to help me impute the null values, I could not impute them easily. However, I did not feel comfortable with simply dropping it, as I found out:

- Among rows where 'Cabin' was **not null**, there were more survivors (~66.7%) than non-survivors (~33.3%).
- Among rows where 'Cabin was **null**, there were more NON-SURVIVORS (~70%) than non-survivors (~30%).
- 53% of the train dataset consists of non-survivors with a null value under 'Cabin'.

Since non-survivors occupy roughly 60% of the dataset overall, we can conclude that non-survivors are likely to also have a null value under 'Cabin' and vice versa, meaning that there is actually somewhat a correlation between having a null value under 'Cabin' and not surviving.

**To resolve the dilemma regarding 'Cabin', I decided to engineer a new column, 'HasCabin', to keep the correlation between having a null 'Cabin' and not survivng as a signal in the data.**

I also conducted the following feature engineering:

1. Rounding values under 'Fare' to 2 decimal places.
2. Creating 'PartySize' by combining 'SibSp' and 'Parch', and adding '1' to include the passenger themselves.
3. 'PartySize' was then further engineered into a binned column called 'PartyBinned', where 'Alone'= PartySize 1, 'Couple' = PartySize 2, 'SmallMedFamily' = PartySize 3-5, 'LargeFamily' = PartySize 6 and above
4. 'Title' reflects each passenger's title. To reduce dimensionality, I decided to group 'Dr.' and 'Rev.' titles under 'Service', and all noble or military titles under 'Rare'. In addition to this, 'Mr.', 'Mrs.', 'Miss.' and 'Master' round out the categories under this column. I engineered this column because I read that the prefix/titles of each passenger not only indicates their class/social status, but also reveals a pattern on who is likely to survive. For example, passengers with the title 'Master' were likely to survive overall. This is because 'Master' refers to the male children of rich parents, who were given the title as they were too young to be 'Misters'.
5. Finally, I dropped 'PassengerId', 'Name', 'Ticket', 'PartySize' and 'Cabin'.

### Test Set
The following differences in test_set required a slightly different course of action:

1. test_set did not have any null values under 'Embarked'.
2. test_set did, however, have one null value under 'Fare', which I imputed manually based on publicly available information.

## Exploratory Data Analysis Findings
1. There are twice as many female than male survivors overall (Figure #1). On the other hand, there are almost 6 times as many male non-survivors than female non-survivors (Figure #2).
![Figure #1](/Users/elimatthewordonez/Dropbox/OTHERS/Data Analytics Training/Personal Projects/Kaggle Titanic Competition/Submissions/Submission 3/Fig. 1.png)
*Figure #1: Sex composition of survivors*

![Figure #2](/Users/elimatthewordonez/Dropbox/OTHERS/Data Analytics Training/Personal Projects/Kaggle Titanic Competition/Submissions/Submission 3/Fig. 2.png)
*Figure #2: Sex composition of non-survivors*

2. Whereas Pclass 1 had the most survivors (Figure #3), Pclass 3 had the most non-survivors (Figure #4).
![Figure #3](/Users/elimatthewordonez/Dropbox/OTHERS/Data Analytics Training/Personal Projects/Kaggle Titanic Competition/Submissions/Submission 3/Fig. 3.png)
*Figure #3: Pclass composition among survivors*

![Figure #4](/Users/elimatthewordonez/Dropbox/OTHERS/Data Analytics Training/Personal Projects/Kaggle Titanic Competition/Submissions/Submission 3/Fig. 4.png)
*Figure #4: Pclass composition among non-survivors*

3. The overwhelming majority of non-survivors had a null 'Cabin' value
![Figure #5](/Users/elimatthewordonez/Dropbox/OTHERS/Data Analytics Training/Personal Projects/Kaggle Titanic Competition/Submissions/Submission 3/Fig. 5.png)
*Figure #5: 'HasCabin' composition among non-survivors*

4. In terms of party size, assengers who travelled alone were least likely to survive.
![Figure #6](/Users/elimatthewordonez/Dropbox/OTHERS/Data Analytics Training/Personal Projects/Kaggle Titanic Competition/Submissions/Submission 3/Fig. 6.png)
*Figure #6: 'PartySize' composition among all observations*

5. In terms of 'Fare', while most paid < $30 to board the Titanic, those who paid more were more likely to survive.
![Figure #7](/Users/elimatthewordonez/Dropbox/OTHERS/Data Analytics Training/Personal Projects/Kaggle Titanic Competition/Submissions/Submission 3/Fig. 7.png)
*Figure #7: 'Fare' composition among all observations*

## Notes on Random Forest Modelling
As stated in the beginning, the final model dataset (for both train and test datasets) was as follows:
|Feature      |                                            Data Dictionary Description                                                   |
|-------------|--------------------------------------------------------------------------------------------------------------------------|
|'Pclass'     |Ticket class. 1=1st, 2=2nd, 3=3rd                                                                                         |
|'Sex'        |Sex (of passenger)                                                                                                        |
|'Age'        |Age of passenger in years                                                                                                 |
|'Fare'       |Passenger fare                                                                                                            |
|'Embarked'   |Port of Embarkation                                                                                                       |
|'HasCabin'   |Indicates whether passenger's 'Cabin' value was null. 1 = not null, 0 = null                                              |
|'PartyBinned'|Size of passenger's party Alone=1 passenger, Couple=2 passenger, SmallMedFamily=3-5 passengers, LargeFamily= >5 passengers|
|'Title'      |Indicates passenger's title                                                                                               |

- I one-hot encoded categorical variables 'Pclass', 'Sex', 'Embarked' and 'PartyBinned'
- I did not scale my numerical features 'Age' and 'Fare', as the model random forest is able to handle them elegantly.

## What This Repository Features

1. My Jupyter Notebook, which includes all my processes from data extraction, to cleaning, engineering, model training and test set treatment.
2. My final dataframe, 'y_test_prediction_df.csv', which includes the predicted 'y' values (in other words, 'Survived') based on the test set, as well as their accompanying 'PassengerId'.
3. This README.md file.