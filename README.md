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
### Exploratory Data Analysis (EDA): 
Visualizations and summary statistics to understand the distribution of contaminants and key features influencing emissions.
### Modeling: 
Implemented classification models to predict whether emissions for each contaminant are likely to increase, decrease, or remain stable in 2023.
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

Fine-tuning models using cross-validation.
Experimenting with ensemble methods to improve accuracy.
Expanding features to include socio-economic data for deeper analysis.

## Acknowledgments
This project was conducted as part of the Machine Learning Analyst diploma program at NorQuest College, with a focus on environmental data analysis and prediction.
