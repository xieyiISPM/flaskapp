from flask import Flask, render_template
from classify import test


app = Flask(__name__)
testData = test()


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/w2v')
def w2v():
    return render_template('w2v.html', testdata=testData)

@app.route('/ohe')
def ohe():
    return render_template('ohe.html')

if __name__ == '__main__':
    app.run(debug=True)
