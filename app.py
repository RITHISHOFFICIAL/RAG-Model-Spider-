
from flask import Flask, render_template, request
from model import answer_question

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    question = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = answer_question(question)
    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
