## LOAN-APPROVAL-PREDICTION-PYTHON


# Overview

This project leverages machine learning to predict loan approval based on historical data. The training dataset, "loan-train.csv," is used to train models, and predictions are made on the "loan-test.csv" dataset.

# Requirements
- Python 3
- Jupyter Notebook
- Libraries: NumPy, Pandas, Matplotlib, Scikit-learn
- Installation

# Installation
1. Clone the repository:

   ```
   https://github.com/RAVINDRA8008/RAVINDRA8008-LOAN-APPROVAL-PREDICTION-PYTHON.git
   ```
2. Navigate to the project directory:

   ```
   cd loan-approval-prediction
   ```
3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```
   
## Code Explanation
The script does the following:
1. Data Loading:

    Loads the training dataset (loan-train.csv) into a Pandas DataFrame.
 2. Data Exploration:

    Displays the first few rows, shape, information, and basic statistics of the dataset.
 3. Handling Missing Values:

    Identifies and fills missing values in the dataset using the mode for categorical variables and the mean for numerical variables.
 4. Data Visualization:

    Generates box plots and histograms to visualize the distribution of various features.
 5. Feature Engineering:

    Creates new features, such as 'TotalIncome' and 'TotalIncome_log', by combining existing features and applying log transformation.
 6. Label Encoding:

    Converts categorical variables into numerical format using label encoding.
 7. Train-Test Split:

    Splits the dataset into training and testing sets for model evaluation.
 8. Model Training (Decision Tree and Naive Bayes):

    Initializes, trains, and fits machine learning models (Decision Tree and Naive Bayes) using the training data.
 9. Model Evaluation:

    Evaluates model performance using accuracy scores on the test set.
    
 11. Predictions on Test Data:

     Processes the test data, applies the trained Naive Bayes model, and makes predictions for loan approval.
# Results
The notebook provides accuracy scores for both the Decision Tree and Naive Bayes models. Additionally, it predicts loan approval for the test data and outputs the results.

# Notes:
1. Make sure you have Git, Python, and Jupyter Notebook installed on your system.
2. If you encounter any issues during the installation or execution, please check error messages and ensure that your environment is properly set up.
