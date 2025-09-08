# Fake News Detector

This project is a simple machine learning model that classifies news as **FAKE** or **REAL**.  
It uses text cleaning, TF-IDF features, and compares two models (Logistic Regression and Passive Aggressive).  
The better model is chosen with cross-validation, then saved for evaluation and a small demo app.

---

## Features
- Cleans text (lowercase, removes URLs and punctuation)  
- Turns text into TF-IDF features (1–2 grams)  
- Compares two models and picks the best one  
- Saves the trained pipeline with `joblib`  
- Confusion matrix + classification report for results  
- Streamlit app for quick predictions  

---

## Project Structure

fake-news-detector/
├── train.py # trains and saves the model
├── evaluate.py # evaluates model + saves confusion_matrix.png
├── app.py # Streamlit app
├── requirements.txt
├── README.md
└── main.ipynb # (optional) notebook I used for exploration


---

## Dataset
The project uses the [Fake News dataset](https://www.kaggle.com/c/fake-news/data).  
Download `news.csv` and place it in the project root folder:

fake-news-detector/
├── train.py
├── evaluate.py
├── app.py
└── news.csv <-- put it here

---

## How to Run

Set up the environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


Train the model
- python train.py --data news.csv
Evaluate it:
- python evaluate.py --data news.csv
Run the demo app:
- streamlit run app.py

Example Output
- Confusion matrix from evaluation:
    Chosen model: PassiveAggressiveClassifier
                precision    recall  f1-score   support

            FAKE     0.9416    0.9446    0.9431       614
            REAL     0.9430    0.9398    0.9414       598

        accuracy                         0.9422      1212
    macro avg     0.9423    0.9422    0.9422      1212
    weighted avg     0.9422    0.9422    0.9422      1212

    Confusion matrix (rows=true, cols=pred):
    [[580  34]
    [ 36 562]]

    Saved model pipeline -> fake_news_detector.joblib

Run the interactive app:
    streamlit run app.py

Paste any news text, and the app will predict FAKE or REAL.
If using Logistic Regression, the app will also display prediction confidences.

Results

Cross-validation model selection between Logistic Regression and Passive Aggressive.

Test set F1-macro ≈ [insert your result here after running train.py].

Balanced evaluation with stratified splits.

Tech Stack

    Python 3.10+
    scikit-learn
    pandas
    matplotlib / seaborn
    Streamlit
    joblib

