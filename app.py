from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('House.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Fetching form data
    Rooms = int(request.form['bedrooms'])
    Bathrooms = int(request.form['bathrooms'])
    Place = int(request.form['location'])
    Area = int(request.form['area'])
    Status = int(request.form['status'])
    Facing = int(request.form['facing'])
    P_Type = int(request.form['type'])
    
    # Input data array
    input_data = np.array([[Rooms, Bathrooms, Place, Area, Status, Facing, P_Type]])
    
    # Predicting the house price
    prediction = model.predict(input_data)[0]
    
    # Returning the prediction to the template
    return render_template('page.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
