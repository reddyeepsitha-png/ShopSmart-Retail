# ShopSmart Retail – Customer Intent Classifier

## Project Overview

ShopSmart Retail is an AI/ML project that classifies customer messages into predefined retail customer-service intents.

The system uses Natural Language Processing (NLP) techniques to analyze customer text and predict the most relevant intent.

## Objectives

* Preprocess customer text data
* Convert text into numerical features using TF-IDF
* Train a machine learning classification model
* Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix
* Save the trained model for reuse
* Build an interactive prediction system for customer messages

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* TF-IDF
* Logistic Regression

## Dataset

The dataset contains customer messages mapped to six customer-service intents:

1. `order_status`
2. `return_request`
3. `refund_enquiry`
4. `delivery_delay`
5. `product_query`
6. `complaint`

The enhanced dataset contains 180 realistic examples with 30 examples per intent.

## Machine Learning Pipeline

The project follows this workflow:

```text
Customer Message
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression
       ↓
Predicted Customer Intent
       ↓
Confidence Score
```

## Model

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts customer messages into numerical feature vectors.

The implementation uses:

* Unigrams and bigrams
* Lowercase text processing
* Sublinear TF scaling

### Logistic Regression

Logistic Regression is used as the classification algorithm.

The model uses balanced class weights to handle differences in class representation.

## Model Evaluation

The enhanced dataset was divided using an 80:20 stratified train-test split.

* Total samples: 180
* Training samples: 144
* Testing samples: 36
* Number of intents: 6
* Test accuracy: **83.33%**

### Classification Performance

| Intent         | Precision | Recall | F1-Score |
| -------------- | --------: | -----: | -------: |
| Complaint      |      0.80 |   0.67 |     0.73 |
| Delivery Delay |      0.67 |   0.67 |     0.67 |
| Order Status   |      0.80 |   0.67 |     0.73 |
| Product Query  |      1.00 |   1.00 |     1.00 |
| Refund Enquiry |      1.00 |   1.00 |     1.00 |
| Return Request |      0.75 |   1.00 |     0.86 |

## Project Files

```text
ShopSmart-Retail/
│
├── aiml_training_data.csv
├── aiml_training_data_original.csv
├── enhanced_training_data.csv
│
├── enhance_dataset.py
├── train_model.py
├── predict.py
│
├── shopsmart_intent_classifier.joblib
├── requirements.txt
└── README.md
```

### File Descriptions

* `aiml_training_data.csv` – Original provided dataset
* `aiml_training_data_original.csv` – Backup of the original dataset
* `enhanced_training_data.csv` – Cleaned and enhanced dataset
* `enhance_dataset.py` – Creates the enhanced dataset
* `train_model.py` – Trains and evaluates the classifier
* `predict.py` – Interactive customer-message prediction system
* `shopsmart_intent_classifier.joblib` – Saved trained model
* `requirements.txt` – Python dependencies
* `README.md` – Project documentation

## How to Run

### 1. Create and activate the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Run the prediction system

```bash
python predict.py
```

Enter a customer message when prompted.

Example:

```text
Customer: Where is my package?

Predicted intent: order_status
Confidence: ...
```

Type `exit` to stop the prediction system.

## Example Intents

| Customer Message                  | Predicted Intent |
| --------------------------------- | ---------------- |
| Where is my package?              | `order_status`   |
| I want to return this item        | `return_request` |
| How can I get a refund?           | `refund_enquiry` |
| My delivery is late               | `delivery_delay` |
| Tell me more about this product   | `product_query`  |
| I have a complaint about my order | `complaint`      |

## Conclusion

The project demonstrates an end-to-end NLP classification workflow for retail customer-service automation. The trained model can classify customer messages into six predefined intents and provide a confidence score for each prediction.

The system provides a foundation for automating customer-support message routing in an e-commerce environment.
