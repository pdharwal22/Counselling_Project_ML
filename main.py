import csv
import os

import firebase_admin
import numpy as np
import requests
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import os
import sys
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestRegressor


from flask import *

cred = credentials.Certificate("serviceAccountKey.json")
# firebase_admin.initialize_app(cred)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceAccountKey.json"
print("Firebase Initialized")

# Add a new document
# db = firestore.Client()
# doc_ref = db.collection(u'users').document(u'alovelace')
# doc_ref.set({
#     u'first': u'Ada',
#     u'last': u'Lovelace',
#     u'born': 1815
# })

# # Then query for documents
# users_ref = db.collection(u'users')

# for doc in users_ref.stream():
#     print(u'{} => {}'.format(doc.id, doc.to_dict()))

app = Flask('MyApp')


@app.route('/')
def index():
    return render_template('index.html')


# # Create new user
# @app.route('/new-user', methods=['POST'])
# def new_user():
#     email = request.form['txtEmail1']
#     password = request.form['txtPassword1']
#     firebase.auth().createUserWithEmailAndPassword(email, password)


# Authenticate existing user
# @app.route('/login', methods=['POST'])
# def authenticate():
#     email = request.form['txtEmail']
#     password = request.form['txtPassword']
#
#     try:
#         auth.sign_in_with_email_and_password(email, password)
#         return render_template('welcome.html')
#
#     except:
#         return 'Invalid credentials. Please try again'


@app.route('/welcome')
def welcome_page():
    return render_template('welcome.html')


@app.route('/add-student')
def register_student():
    return render_template('add-student.html')


# Add student to the firebase
@app.route('/register-student', methods=['POST', 'GET'])
def register():
    db = firestore.Client()
    doc_ref = db.collection(u'students').document(request.form['txtRoll'])

    doc_ref.set({
        u'name': request.form['txtName'],
        u'phone': request.form['txtPhone'],
        u'reg': request.form['txtRoll'],
        u'email': request.form['txtEmail'],
        u'house': request.form['txtHouse'],
        u'street': request.form['txtStreet'],
        u'city': request.form['txtCity'],
        u'state': request.form['txtState']
    })

    return render_template('welcome.html')


# Fetch all students from the firebase
@app.route('/view-student')
def view_student():
    # fetch data from firebase
    db = firestore.Client()
    docs = db.collection(u'students').stream()
    result = []
    for doc in docs:
        result.append(doc.to_dict())
        print(doc.to_dict())

    return render_template('view-student.html', result=result)


# View particular student by applying where condition
@app.route('/stu-det')
def detail():
    return render_template('detail-reg.html')


@app.route('/student-detail', methods=['POST'])
def student_details():
    # fetch data from firebase
    db = firestore.Client()
    docs = db.collection(u'students').where('reg', '==', request.form['txtReg2']).get()
    result = []
    for doc in docs:
        result.append(doc.to_dict())
        print(doc.to_dict())

    return render_template('student-detail.html', result=result)


# Update student details
@app.route('/update')
def update():
    return render_template('update-student.html')


@app.route('/update-student', methods=['POST', 'GET'])
def update_student():
    db = firestore.Client()
    doc_ref = db.collection(u'students').document(request.form['txtRoll'])

    doc_ref.set({
        u'name': request.form['txtName'],
        u'phone': request.form['txtPhone'],
        u'reg': request.form['txtRoll'],
        u'email': request.form['txtEmail'],
        u'house': request.form['txtHouse'],
        u'street': request.form['txtStreet'],
        u'city': request.form['txtCity'],
        u'state': request.form['txtState']
    })

    print('Data updated successfully.')
    return render_template('view-student.html')


# Delete student from firebase
@app.route('/delete', methods=['GET'])
def delete_student():
    db = firestore.Client()
    doc_ref = db.collection('students').document(request.form['txtRoll']).get()
    doc_ref.delete()

    return 'Student Deleted successfully'


@app.route('/marks')
def marks_page():
    return render_template('add-marks.html')


# Add marks to the firebase
@app.route('/add-marks', methods=['POST'])
def add_marks():
    db = firestore.Client()
    doc_ref = db.collection(u'students').document(request.form['txtReg']).collection('marks').document(
        request.form['txtSubject'])

    doc_ref.set({
        u'reg1': request.form['txtReg'],
        u'subject': request.form['txtSubject'],
        u'c1': request.form['c1'],
        u'c2': request.form['c2'],
        u'c3': request.form['c3'],
        u'c4': request.form['c4'],
        u'c5': request.form['c5'],
        u'c6': request.form['c6'],
        u'c7': request.form['c7'],
        u'c8': request.form['c8'],
        u'c9': request.form['c9']
    })

    marks_ref = db.collection(u'marks')
    for doc in marks_ref.stream():
        print(u'{} => {}'.format(doc.id, doc.to_dict()))

    return render_template('add-marks.html')


# View marks of the students
@app.route('/mark-det')
def marks():
    return render_template('detail-marks.html')


@app.route('/view-marks', methods=['POST', 'GET'])
def view_marks():
    # fetch data from firebase
    db = firestore.Client()
    doc = db.collection('students').document(request.form['txtReg3']).collection('marks').document(request.form['txtSub']).get()
    print(doc.to_dict())

    # doc.to_csv('Marks.csv')
    return render_template('view-marks.html', result=doc.to_dict())


@app.route('/export')
def export_page():
    return render_template('export-marks.html')


# Exporting data from firebase into csv
@app.route('/export-marks', methods=['POST'])
def export_marks():
    db = firestore.Client()
    doc = db.collection('students').document(request.form['txtReg4']).collection('marks').document(request.form['txtSub1']).get()
    data = doc.to_dict()

    file = open('Marks.csv', 'a')
    file.write("{},{},{},{},{},{},{},{},{},{},{}".format(data["reg1"], data['subject'], data['c1'], data['c2'], data['c3'], data['c4'], data['c5'],
                                     data['c6'], data['c7'], data['c8'], data['c9'], '\n'))
    file.write('\n')

    return render_template('export-marks.html')


@app.route('/train')
def train():
    return render_template('train-page.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    reg5 = request.form['txtReg6']
    sub2 = request.form['txtSub2']

    if sub2 == 'english' or sub2 == 'English':
        df = pd.read_csv('Datasets/eng.csv')
    elif sub2 == 'maths' or sub2 == 'Maths':
        df = pd.read_csv('Datasets/maths.csv')
    elif sub2 == 'science' or sub2 == 'Science':
        df = pd.read_csv('Datasets/science.csv')
    elif sub2 == 'social science' or sub2 == 'Social Science':
        df = pd.read_csv('Datasets/socio.csv')
    elif sub2 == 'g.k.' or sub2 == 'G.K.':
        df = pd.read_csv('Datasets/gk.csv')
    elif sub2 == 'physical education' or sub2 == 'Physical Education':
        df = pd.read_csv('Datasets/phed.csv')
    else:
        return 'Wrong Input! Please enter the correct value'

    X = np.arange(1, 10)
    X = X.reshape((9, 1))

    Y = df.loc[:, [reg5]]
    Y = np.array(Y)

    x_train = X[0:10:2]
    x_test = X[1:10:2]

    y_train = Y[0:10:2]
    y_test = Y[1:10:2]

    reg = RandomForestRegressor(n_estimators=60, max_depth=30, n_jobs=-1, warm_start=True)
    reg.fit(x_train, y_train)

    Y_pred = reg.predict(np.array([[10]]))
    print('Y_pred:')
    print(Y_pred)
    return render_template('predict-result.html', message='Marks in subject {} in class 10 is: {}'.format(sub2, Y_pred))


if __name__ == '__main__':
    app.run()
