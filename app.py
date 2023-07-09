import pickle
import datetime
import numpy as np
from cs50 import SQL
from flask import Flask, request, render_template

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

# Load the trained model from a pickle file
model = pickle.load(open("models/model.pkl", "rb"))

# Create a connection to the SQLite database
db = SQL("sqlite:///cardio.db")


@app.route("/")
def home():
    # Render the home page template
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # If the request method is GET, render the prediction form
        return render_template("predict.html")
    else:
        # If the request method is POST, process the form data and make a prediction
        # Retrieve the form data and convert it to a list of float values
        features = [float(value) for value in request.form.values()]
        
        # Convert the features list to a numpy array and reshape it for the model input
        features_format = np.array(features).reshape(1, -1)
        
        # Make a prediction using the loaded model
        prediction = model.predict(features_format)

        if prediction == 1:
            analysis = "Positive"
        else:
            analysis = "Negative"

        # Get the current date and time
        date = datetime.datetime.now()
        
        # Insert the prediction and date into the database
        db.execute("INSERT INTO users (analysis, date) VALUES (?, ?)", analysis, date)

        # Render the results page template with the prediction value
        return render_template("results.html", prediction=prediction)


@app.route("/history")
def history():
    # Retrieve all data from the database table, ordered by date in descending order
    data = db.execute("SELECT * FROM users ORDER BY date DESC")
    
    # Render the history page template with the retrieved data
    return render_template("history.html", data=data)


if __name__ == "__main__":
    # Run the Flask application in debug mode
    app.run(debug=True)
