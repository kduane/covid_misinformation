# imports
import numpy as np
import pickle
from flask import Flask, Response, request, render_template, jsonify
# initialize the flask app
app = Flask('my_app')
# route 1: hello world
@app.route('/')
# return a simple string
def home():
    return 'Thanks for checking out our misinformation classifier!'
# route 2: return a 'web page'


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
    X_test = np.array([
        int(data['OverallQual']),
        int(data['FullBath']),
        int(data['GarageArea']),
        int(data['LotArea'])
    ]).reshape(1, -1) #turns [1,2,3,4] into [[1,2,3,4]]

    model = pickle.load(open('assets/model.p', 'rb'))
    pred = model.predict(X_test) #[1134]
    pred = round(pred[0], 2)
    return render_template('results.html', prediction = pred)


# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # if we run 'python app_starter.py' in the terminal
    app.run(debug = True)