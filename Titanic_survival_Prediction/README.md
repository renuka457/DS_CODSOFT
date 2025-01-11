# Titanic Survival Prediction

This project aims to predict the survival status of passengers aboard the Titanic. The dataset contains information such as the passenger's age, gender, class, and more. Using this data, we build a machine learning model to predict whether a passenger survived or not.

## Dataset
The dataset is from Kaggle's Titanic dataset and can be found [here](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

### Features
- `PassengerId`: Unique ID for each passenger
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings or spouses aboard the Titanic
- `Parch`: Number of parents or children aboard the Titanic
- `Fare`: Fare paid by the passenger
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- `Survived`: 1 if the passenger survived, 0 if not (target variable)

### Steps
1. **Data Preprocessing**:
   - Load and clean the data
   - Handle missing values
   - Encode categorical features (like Sex, Embarked)

2. **Exploratory Data Analysis (EDA)**:
   - Visualize data distribution
   - Check correlations between features and survival

3. **Modeling**:
   - Train a Logistic Regression model to predict survival
   - Evaluate the model using accuracy score

### Results
- Accuracy: ~80%
- Visualizations: [Provide plots like histograms, bar charts]

