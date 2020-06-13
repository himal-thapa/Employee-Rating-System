from flask import Flask, request, url_for, render_template
import pickle as pickle

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        attendance = request.form['attendance']
        feedback = request.form['feedback']
        overtime = request.form['overtime']
        education = request.form['education']
        flexibility = request.form['flexibility']
        punctuality = request.form['punctuality']
        teamwork  = request.form['teamwork']
        presentationskill  = request.form['presentationskill']
        meetsdeadline = request.form['meetsdeadline']

        svmModel = pickle.load(open("svm.model", "rb"))
        randomForestModel = pickle.load(open("randomForest.model", "rb"))
        naiveModel = pickle.load(open("naive.model", "rb"))
        scalerX = pickle.load(open("scalerX", "rb"))

        to_predict = [attendance, feedback, overtime, education, flexibility, punctuality, teamwork, presentationskill, meetsdeadline]

        svm_result = svmModel.predict(scalerX.transform([to_predict]))
        randomForest_result = randomForestModel.predict(scalerX.transform([to_predict]))
        naive_result = naiveModel.predict(scalerX.transform([to_predict]))

        return render_template("result.html", svm_result=svm_result[0], randomForest_result=randomForest_result[0], naive_result=naive_result[0] , to_predict=to_predict)

if __name__ == "__main__":
    app.run(debug=True)