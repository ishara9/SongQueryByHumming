from flask import Flask, render_template, request
from flask_cors import CORS

import os
import json

from index import search_tune

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# range = cl.bpm_range
# pattern = data_preprocessor.path

cors = CORS(app, resource={
    r"/*": {
        "origins": "*"
    }
})


@app.route('/')
def app_route():
    return render_template('index.html', name=range)


@app.route('/echo', methods=['GET', 'POST'])
def echo():
    return "{\"response\": \"Song Supported!\"}"


@app.route('/pattern', methods=['GET', 'POST'])
def generate_patter():
    if request.method == 'POST':
        json_request = request.get_json()

        return "{}"
    return "Unsupported Request"


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['data']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        _list = search_tune()
        return json.dumps({'songList': _list})


if __name__ == '__main__':
    app.run(debug=True)
