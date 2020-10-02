# imports
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask import Flask, Response, request, render_template, jsonify
# initialize the flask app
app = Flask('my_app')

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
    cvec = CountVectorizer(min_df = 2, max_features = 5000, ngram_range = (1,2), stop_words= None)
    X_test = cvec.fit_transform(data['text']).reshape(1, -1) #turns [1,2,3,4] into [[1,2,3,4]]
    r_dict = {
        1: "likely misinformation, proceed with caution.",
        2: "likely valid."
    }
    model = pickle.load(open('assets/model.p', 'rb'))
    pred = model.predict(X_test) # 0 or 1
    pred = r_dict[pred]
    return render_template('results.html', prediction = pred)


# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # if we run 'python app_starter.py' in the terminal
    app.run(debug = True)