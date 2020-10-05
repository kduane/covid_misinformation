# imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask import Flask, Response, request, render_template, jsonify
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

# initialize the flask app
app = Flask('my_app')

df = pd.read_csv('./datasets/combined_data.csv')
df = df.drop(columns = 'Unnamed: 0')
df['label'] = df['label'].map({'true' : 0,'false' : 1,'misleading' : 1})
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)
estimator = []
estimator.append(('gs_bag', BaggingClassifier()))
estimator.append(('mnnb', MultinomialNB(alpha = 0.1)))
estimator.append(('lrcv', LogisticRegressionCV(penalty = 'l2', solver = 'liblinear')))
estimator.append(('rfc', RandomForestClassifier(bootstrap = True, max_depth = 50, min_samples_leaf = 2, min_samples_split = 5)))
cvec = CountVectorizer(min_df = 2, max_features = 5000, ngram_range = (1,2), stop_words= None)
Z_train = cvec.fit_transform(X_train)
Z_test = cvec.transform(X_test)
vote = VotingClassifier(estimators = estimator, voting = 'soft')
vote.fit(Z_train, y_train)

@app.route('/')

def home():
    return 'Thanks for checking out our misinformation classifier!'

# route 1: show a form to the user
@app.route('/form')
def form():
# use flask's render_template function to display an html page
    return render_template('form.html')

# route 2: accept the form submission and do something fancy with it
@app.route('/submit')
def submit():
    # load in the form data from the incoming request
    data = request.args # form data
    # manipulate data into a format that we pass to our model    
    input_to_vectorize = [data['text']]
    X_test = cvec.transform(input_to_vectorize) #.reshape(1, -1) #turns [1,2,3,4] into [[1,2,3,4]]
    print(X_test)
    r_dict = {
        0 : "likely valid.",
        1 : "likely misinformations."
    }
    model = pickle.load(open('assets/model.p', 'rb'))
    pred = model.predict(X_test) # 0 or 1
    print(pred)
    pred = r_dict[pred[0]]
    return render_template('result.html', prediction = pred)


# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # if we run 'python app_starter.py' in the terminal
    app.run(debug = True)