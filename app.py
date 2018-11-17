from flask import Flask, render_template
from classify import oheClassify


app = Flask(__name__)
#oheAccuracy = oheClassify()
w2vAccuracy = oheClassify()


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/w2v')
def w2v():
    return render_template('w2v.html', w2vscore=w2vAccuracy)

@app.route('/ohe')
def ohe():
    return render_template('ohe.html')

if __name__ == '__main__':
    app.run(debug=True)
