Customer Churn Prediction Using Machine Learning

📌 Project Overview

This project predicts whether a telecom customer is likely to churn (leave the company) or stay using machine learning classification algorithms.

The project covers the complete machine learning workflow:

Data Loading → Data Cleaning → Encoding → Train/Test Split → Feature Scaling → Model Training → Evaluation → Model Comparison → Prediction

⸻

🎯 Objective

The main objective is to build a classification system that predicts:

* 0 → Customer will not churn
* 1 → Customer will churn

The project compares five different machine learning models and evaluates them using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

⸻

📂 Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges
* Churn

The customerID column is removed because it is an identifier and is not useful for prediction.

⸻

🛠️ Technologies & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

⸻

🔄 Data Preprocessing

1. Data Loading

The dataset is loaded using Pandas.

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

2. Handling Missing Values

The TotalCharges column contains blank values. These values are handled before training the machine learning models.

After cleaning, the dataset contains 7,032 records.

3. Removing Customer ID

The customerID column is removed because it does not contribute meaningful information to the prediction.

df = df.drop("customerID", axis=1)

4. Encoding Categorical Variables

Categorical features are converted into numerical values using One-Hot Encoding.

pd.get_dummies(df, drop_first=True)

5. Train-Test Split

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

random_state=42 and stratify=y are used.

6. Feature Scaling

StandardScaler is used to standardize the features before applying models that are sensitive to feature scale.

⸻

🤖 Machine Learning Models

The following five classification algorithms are trained and compared:

1. Logistic Regression

A linear classification algorithm used to predict whether a customer will churn.

2. K-Nearest Neighbors (KNN)

Classifies a customer based on the classes of nearby data points.

3. Naive Bayes

A probabilistic classification algorithm based on Bayes’ theorem.

4. Decision Tree

Uses a tree-like structure of decision rules to classify customers.

5. Support Vector Machine (SVM)

Finds an optimal decision boundary between different classes.

⸻

📊 Model Performance

The models were evaluated using the test dataset.

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	80.38%	64.76%	57.49%	60.91%
KNN	75.27%	53.57%	52.14%	52.85%
Naive Bayes	64.46%	41.84%	86.36%	56.37%
Decision Tree	70.79%	45.14%	45.99%	45.56%
SVM	78.68%	62.59%	49.20%	55.09%

⸻

🏆 Best Performing Model

Based on the overall evaluation results, Logistic Regression performs the best among the tested models.

Logistic Regression Results

* Accuracy: 80.38%
* Precision: 64.76%
* Recall: 57.49%
* F1-Score: 60.91%

Logistic Regression provides the highest overall accuracy and F1-score in this project.

⸻

🔲 Confusion Matrices

Logistic Regression

[[916 117]
 [159 215]]

KNN

[[864 169]
 [179 195]]

Naive Bayes

[[584 449]
 [ 51 323]]

Decision Tree

[[833 200]
 [204 170]]

SVM

[[923 110]
 [190 184]]

⸻

📈 Key Observations

* Logistic Regression achieved the highest accuracy: 80.38%.
* Logistic Regression also achieved the highest F1-score: 60.91%.
* Naive Bayes achieved the highest recall: 86.36%.
* KNN achieved 75.27% accuracy.
* Decision Tree achieved 70.79% accuracy.
* SVM achieved 78.68% accuracy.

The best model can depend on the business requirement. If the goal is overall performance, Logistic Regression is the strongest model in this experiment. If identifying as many actual churn customers as possible is the priority, Naive Bayes has the highest recall.

⸻

🔄 Project Workflow

Telco Customer Churn Dataset
            ↓
       Data Loading
            ↓
      Data Exploration
            ↓
       Data Cleaning
            ↓
    Handle Missing Values
            ↓
     Remove Customer ID
            ↓
      One-Hot Encoding
            ↓
      Train/Test Split
            ↓
      Feature Scaling
            ↓
 ┌──────────┼──────────┬──────────┬──────────┐
 ↓          ↓          ↓          ↓          ↓
Logistic    KNN     Naive Bayes  Decision    SVM
Regression                     Tree
 └──────────┼──────────┴──────────┴──────────┘
            ↓
      Model Evaluation
            ↓
Accuracy / Precision / Recall / F1
            ↓
       Model Comparison
            ↓
     Best Model Selection
            ↓
      Customer Prediction

⸻

🚀 How to Run the Project

1. Clone the Repository

git clone <your-repository-url>

2. Open the Project Folder

cd <project-folder>

3. Install Required Libraries

pip install numpy pandas matplotlib seaborn scikit-learn jupyter

4. Start Jupyter Notebook

jupyter notebook

5. Run the Notebook

Open:

main(4).ipynb

Make sure the dataset CSV file is available in the correct project directory.

⸻

📁 Project Structure

Customer-Churn-Prediction/
│
├── main(4).ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── requirements.txt

⸻

🔮 Future Improvements

The project can be improved by:

* Hyperparameter tuning
* Cross-validation
* Feature selection
* Handling class imbalance
* ROC-AUC comparison
* Improving recall for churn customers
* Saving the trained model
* Creating a Streamlit web application
* Deploying the model as an API

⸻

👨‍💻 Author

Shantanu Gaikwad

⸻

📌 Conclusion

This project demonstrates a complete Customer Churn Prediction machine learning pipeline, starting from data preprocessing and feature engineering to model training, evaluation, comparison, and prediction.

Among the five tested classification algorithms, Logistic Regression achieved the best overall performance with 80.38% accuracy and a 60.91% F1-score.
