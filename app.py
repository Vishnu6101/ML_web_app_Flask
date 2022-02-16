from flask import Flask, render_template,request, url_for, redirect

import linearRegression as LinReg
import logisticRegression as LogReg

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/linear', methods=['GET', 'POST'])
def linearRegression():
    if request.method == 'POST':
        Alpha = request.form['lambda']
        return redirect(url_for('lambdaValue', alpha=Alpha))
    else:
        X = LinReg.Ridge()
        return render_template("linear_reg.html", coeff=X[0], inter=X[1], rmse=X[2], mse=X[3], mae=X[4], R2=X[5])


@app.route('/linear/<alpha>', methods=['GET', 'POST'])
def lambdaValue(alpha):
    X = LinReg.Ridge(alpha)
    return render_template("linear_reg.html", coeff=X[0], inter=X[1], rmse=X[2], mse=X[3], mae=X[4], R2=X[5])


@app.route('/logistic', methods=['GET', 'POST'])
def logisticRegression():
    if request.method == 'POST':
        C = request.form['C']
        return redirect(url_for('cValue', cVal=C))
    else:
        Y = LogReg.LogRegwithC()
        return render_template('logistic_reg.html', coeff=Y[0], inter=Y[1], accuracy=Y[2], precision=Y[3], recall=Y[4], 
                            F_Score=Y[5], tp=Y[6], fp=Y[7], fn=Y[8], tn=Y[9], total=Y[10])


@app.route('/logistic/<cVal>', methods=['GET', 'POST'])
def cValue(cVal):
    Y = LogReg.LogRegwithC(cVal)
    return render_template('logistic_reg.html', coeff=Y[0], inter=Y[1], accuracy=Y[2], precision=Y[3], recall=Y[4], 
                            F_Score=Y[5], tp=Y[6], fp=Y[7], fn=Y[8], tn=Y[9], total=Y[10])