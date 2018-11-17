from flask import Flask, render_template
from classify import oheClassify,w2vClassify


app = Flask(__name__)
#oheAccuracy = oheClassify()



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/w2v', methods=['GET'])
def w2v():
    w2vAccuracy = w2vClassify()
    return render_template('w2v.html', w2vscore=w2vAccuracy)

@app.route('/ohe')
def ohe():
    oheAccuracy = oheClassify()
    return render_template('ohe.html',ohescore=oheAccuracy)

if __name__ == '__main__':
    app.run(debug=True)
