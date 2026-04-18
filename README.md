# Prediciting NYC motor vehicle collision severity
Myra Mulongoti
May 3 2026

## Project Description
This project studies a multiclass classification problem using the NYC Open Data Motor Vehicle Collisions dataset. The goal is to predict crash severity from crash information. 

The target variable is a severity class with three possible values:
        - **None**: no injuries and no deaths
        - **Injury**: at least one injury and no deaths
        - **Fatal**: at least one death

The project compares the performance of two ML models:
    1) Logistic Rgression
    2) Random Forest Classifier

## Link to the entire dataset: 
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data

Only a random sample of 150000 rows is used for the project. 


## Structure of the directory:
- `code/` : all implementaiton code including data laoding, preprocessing, model implementation etc
- `figures/`: any charts or visualizations created for the report and outputs from training
- `reports/`: contains the project report and the project proposal in pdf format

## Implementation and notes on how to run

### Dependencies
- Requires latest version of python and the libraries listed in requirements.txt
- To install the requirements run: 
    `pip install -r requirements.txt`

### Make dataset
- To run the dataset fetching, cleaning, and preprocessing, use the following commands:
    - Fetch random sample of 100000 rows: `python code/fetch_data.py`
    - Cleaned dataset and build features, save: `python code/make_dataset.py`

### Train/evalaute models;
 All model scripts:
   - load `Xraw`, `y`, 
      - split into train/test split
      - fit the preprocessing pipeline only on the training set (preprocessing pipeline defined in `code/preprocessing.py`)
      - transform train/test split
      - train the model and compute metrics 
      - write report files to `figures/model_reports`

To train models and make predictions use the commands:
    `python code/random_forest.py`
    `python code/logistic_regression.py`

### Visualizations and Charts
To create charts/visualizations use the following commans:
    `python code/make_charts.py`

The following charts are generated and saved to `figures/`: 
    - `class_distr.png` 
    - `bar_most_common_contr_factors.png`
    - `model_metric_comparison.png`
    - `rf_per_class_metrics.png`
    - `lr_per_class_metrics.png`
    - `rf_confusion_matrix.png`
    - `lr_confusion_matrix.png`