# National Pollutant Release Inventory (NPRI) Classification Models for Criteria Air Contaminants
## Project Overview
This project aims to predict trends in the release of criteria air contaminants in Canada using data from the National Pollutant Release Inventory (NPRI). Criteria air contaminants (CACs) include pollutants like sulfur dioxide, nitrogen oxides, volatile organic compounds, particulate matter, and carbon monoxide. These contaminants impact air quality and human health, making it crucial to understand and predict their trends.

# Dataset
The dataset used in this project is sourced from the National Pollutant Release Inventory (NPRI), which tracks pollutants released to air, water, and land in Canada. The dataset includes information on emissions of various contaminants across multiple industries and regions, along with other environmental variables.

# Target Variables: 
The model focuses on predicting the trends for the following CACs:

Sulphur Dioxide (SO₂)
Nitrogen Oxides (NOₓ)
Volatile Organic Compounds (VOCs)
Particulate Matter (PM)
Carbon Monoxide (CO)
# Features:
Key features include geographic location, industry classification, reported amounts of emissions, and year of data reporting, among other environmental and industry-specific factors.

# Project Structure
## Data Preprocessing: 
Initial steps include data cleaning, handling missing values, and normalizing data where necessary.
## Exploratory Data Analysis (EDA): 
Visualizations and summary statistics to understand the distribution of contaminants and key features influencing emissions.
## Modeling: 
Implemented classification models to predict whether emissions for each contaminant are likely to increase, decrease, or remain stable in 2023.
# Model Evaluation: 
Accuracy, F1 score, and confusion matrix are used to assess model performance.

# Classification Models
The following machine learning algorithms were explored for classification:

Decision Tree Classifier
Random Forest Classifier
Logistic Regression
Support Vector Machine (SVM)

# Results
Achieved the highest accuracy with *Random Forest Classifier*.

# Dependencies
Python 3.8+
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

# Results and Visualizations
Visualizations were created to represent the emission trends of CACs by region and industry, showing insights into high-pollution areas and industry-specific emissions.

# Future Work
Possible improvements could include:

Fine-tuning models using cross-validation.
Experimenting with ensemble methods to improve accuracy.
Expanding features to include socio-economic data for deeper analysis.

# Acknowledgments
This project was conducted as part of the Machine Learning Analyst diploma program at NorQuest College, with a focus on environmental data analysis and prediction.
