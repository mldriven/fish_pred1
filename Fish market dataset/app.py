from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('fish_weight_model.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get data from form
        data = request.form.to_dict()
        # Check if 'Species' is in the form data
        if 'Species' not in data:
            return "Error: No species data provided. Please include an input field named 'Species'."
        # Use data to make prediction with your model
        result = model.predict(data)
        # Convert the result to a string so it can be displayed in the HTML
        result_str = str(result)
        return render_template('results.html', result=result_str)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

