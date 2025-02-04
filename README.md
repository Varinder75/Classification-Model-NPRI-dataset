# National Pollutant Release Inventory (NPRI) Classification Models for Criteria Air Contaminants
## Project Overview
This project aims to predict trends in the release of criteria air contaminants (CACs) in Canada using data from the National Pollutant Release Inventory (NPRI). These contaminants include sulfur dioxide (SO₂), nitrogen oxides (NOₓ), volatile organic compounds (VOCs), particulate matter (PM), and carbon monoxide (CO). The goal is to predict whether emissions of these pollutants will increase, decrease, or remain stable in 2023, which is crucial for regulatory and environmental decision-making.

## Dataset
The dataset used in this project is sourced from the National Pollutant Release Inventory (NPRI), which tracks pollutants released to air, water, and land in Canada. It contains emissions data across multiple industries and regions, as well as other environmental variables.

## Key Dataset Features:
•Year: Year of data reporting.
•Region: Geographic location (e.g., province, city).
•Industry Classification: Type of industry (e.g., manufacturing, agriculture).
•Pollutants: Emission amounts of pollutants (SO₂, NOₓ, VOCs, PM, CO).
•Emissions: Amount of pollutants released (in metric tons).
•Emission Change: Difference in emissions between 2022 and 2023.
## Target Variables: 
The model focuses on predicting the trends for the following CACs:

Sulphur Dioxide (SO₂)
Nitrogen Oxides (NOₓ)
Volatile Organic Compounds (VOCs)
Particulate Matter (PM)
Carbon Monoxide (CO)

## Features:
Key features include geographic location, industry classification, reported amounts of emissions, and year of data reporting, among other environmental and industry-specific factors.

## Project Structure
### Data Preprocessing: 
Initial steps include data cleaning, handling missing values, and normalizing data where necessary.
1. **Loading Data:**
   Load the dataset into a Pandas DataFrame.

   ```python
   import pandas as pd
   data = pd.read_csv('npri_data.csv')
   print(data.head())
   ```
2. **Handling missing values:**
   Handle missing values using imputation or dropping rows.
   ```python
   data.fillna(data.mean(), inplace=True)
   ```
3. **Feature engineering:**
   ```python
   # Select relevant columns
   releases_filtered = releases_df[['Reporting_Year / Année', 'Substance Name (English) / Nom de substance (Anglais)',
                                 'Number of employees',
                                 'Air_Releases', 'Land_Releases', 'Water_Releases',
                                 'Total_Releases'
                                      ]]
   # Melt the DataFrame to long format
   releases_long = releases_filtered.melt(
   id_vars=['Reporting_Year / Année', 'Substance Name (English) / Nom de substance (Anglais)',
             'Number of employees'],
   var_name='Release_Type',
    value_name='Release'
   )
   # Convert the year to datetime format
   releases_long['Reporting_Year / Année'] = pd.to_datetime(releases_long['Reporting_Year / Année'], format='%Y')

   # Inspect the transformed DataFrame
   print(releases_long.head())
   # Convert the year to datetime format
   releases_long['Reporting_Year / Année'] = pd.to_datetime(releases_long['Reporting_Year / Année'], format='%Y')
   # Inspect the transformed DataFrame
   print(releases_long.head())
   ```
### Exploratory Data Analysis (EDA): 
Visualizations and summary statistics to understand the distribution of contaminants and key features influencing emissions.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing emission distribution by year
sns.boxplot(x='Year', y='Emissions', data=data)
plt.title("Emission Distribution by Year")
plt.show()
```

### Modeling: 
Implemented classification models to predict whether emissions for each contaminant are likely to increase, decrease, or remain stable in 2023.
```python
def process_substance(substance_name, substance_data):
    # Step 1: Extract data for the given substance
    data = substance_data[substance_name]

    # Step 2: Prepare the features and target variable
    X = data[['Lag_1', 'Lag_2', 'Lag_3', 'Number of employees']]

    # Target variable
    y = data['Release_Category']

    # Combine X and y for shuffling
    combined_data = pd.concat([X, y], axis=1)

    # Step 3: Drop rows with any NaNs in features or target
    combined_data = combined_data.dropna()

    # Check if there are still any NaNs in the target variable
    if combined_data['Release_Category'].isnull().any():
        print(f"Warning: Found NaN values in {substance_name}'s target variable after filtering. Aborting processing.")
        return None, None  # Return None if NaN is found

    # Separate back into features and target variable after cleaning
    X_clean = combined_data.drop('Release_Category', axis=1)
    y_clean = combined_data['Release_Category']

    # Encode categorical variables
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_clean)

    # Shuffle the combined dataset
    combined_data_shuffled = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the data into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(combined_data_shuffled))
    train_data = combined_data_shuffled[:train_size]
    test_data = combined_data_shuffled[train_size:]

    # Further split the training set into training and validation sets (80% train, 20% validation)
    val_size = int(0.2 * len(train_data))
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]

    # Separate the features and target variable for each set
    X_train = train_data.drop('Release_Category', axis=1)
    y_train = train_data['Release_Category']
    X_val = val_data.drop('Release_Category', axis=1)
    y_val = val_data['Release_Category']
    X_test = test_data.drop('Release_Category', axis=1)
    y_test = test_data['Release_Category']

    # Define the model
    model = RandomForestClassifier(random_state=42)

    # Set up the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform Randomized Search
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                       n_iter=10, scoring='f1_weighted', cv=3, verbose=2, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Get the best parameters and estimator
    best_model = random_search.best_estimator_

    # Make predictions with the best model for validation phase
    y_val_pred_best = best_model.predict(X_val)

    # Evaluate the model on the validation set
    print(f"Validation Classification Report for {substance_name}:")
    print(classification_report(y_val, y_val_pred_best))

    print(f"Validation Confusion Matrix for {substance_name}:")
    print(confusion_matrix(y_val, y_val_pred_best))

    # Make predictions on the test set to evaluate the final model
    y_test_pred_best = best_model.predict(X_test)

    # Evaluate the model on the test set
    print(f"Test Classification Report for {substance_name}:")
    print(classification_report(y_test, y_test_pred_best))

    print(f"Test Confusion Matrix for {substance_name}:")
    print(confusion_matrix(y_test, y_test_pred_best))

    return best_model, random_search.best_params_

# Now, call the function for each substance
models = {}
for substance in target_substances:
    print(f"Processing {substance}...")
    model, best_params = process_substance(substance, substance_data)
    models[substance] = model  # Store the trained model for later use

    print(f"Best parameters for {substance}: {best_params}")
    print("------------------------------------------------------------")

```
### Feature importance:
We can evaluate the feature importance in our random forest model:
```python
# Get feature importance
importances = model.feature_importances_

# Visualize the importance
plt.bar(X.columns, importances)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

### Model Evaluation: 
Accuracy, F1 score, and confusion matrix are used to assess model performance.

## Classification Models
The following machine learning algorithms were explored for classification:

Decision Tree Classifier
Random Forest Classifier
Logistic Regression
Support Vector Machine (SVM)

## Results
Achieved the highest accuracy with *Random Forest Classifier*.

## Dependencies
Python 3.8+
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Results and Visualizations
Visualizations were created to represent the emission trends of CACs by region and industry, showing insights into high-pollution areas and industry-specific emissions.

## Future Work
Possible improvements could include:

•Fine-tuning models using cross-validation.
•Experimenting with ensemble methods to improve accuracy.
•Expanding features to include socio-economic data for deeper analysis.

## Acknowledgments
This project was conducted as part of the Machine Learning Analyst diploma program at NorQuest College, with a focus on environmental data analysis and prediction.
