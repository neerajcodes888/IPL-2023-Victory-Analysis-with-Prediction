from flask import Flask, render_template, request
import joblib
import pandas as pd

model = joblib.load('IPL_Prediction_Model.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

def calculate_crr(target_runs, runs_left, balls_left):
    total_runs = target_runs - runs_left
    total_overs = balls_left / 6
    return total_runs / total_overs

def calculate_rrr(target_runs, runs_left, balls_left):
    remaining_runs = target_runs - runs_left
    remaining_overs = balls_left / 6
    return remaining_runs / remaining_overs


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        city = request.form.get('city')
        runs_left = float(request.form.get('runs_left'))
        balls_left = float(request.form.get('balls_left'))
        wickets_left = float(request.form.get('wickets_left'))
        target = float(request.form.get('target'))
        
        current_run_rate = calculate_crr(target,runs_left,balls_left)
        required_run_rate = calculate_rrr(target, runs_left,balls_left)

        # Create a DataFrame with the input values
        data = [[batting_team, bowling_team, city, runs_left, balls_left, wickets_left,
             current_run_rate, required_run_rate, target]]
        columns = ['BattingTeam', 'BowlingTeam', 'City', 'runs_left', 'balls_left',
               'wickets_left', 'current_run_rate', 'required_run_rate', 'target']
        input_df = pd.DataFrame(data, columns=columns)

        team1 = batting_team
        team2 = bowling_team

        # Make the prediction using the loaded model
        prediction = model.predict_proba(input_df)

        return render_template('prediction.html',
                           team1=team1,
                           team2=team2,
                           probability1=int(prediction[0, 0] * 100),
                           probability2=int(prediction[0, 1] * 100))
 

    else:
        return render_template("prediction.html") 


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
