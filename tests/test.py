from flask import Flask, request

app = Flask(__name__)


@app.route('/query-example')
def query_example():
    language = request.args.get('language')
    framework = request.args.get('framework')
    return '<h1>The language value is: {} and the framework is {}!</h1>'.format(language, framework)


@app.route('/form-example', methods=['GET', 'POST'])
def form_expample():
    if request.method == 'POST':
        language = request.form.get('language')
        framework = request.form.get('framework')

        return '''<h1>The language value is: {}</h1>
        <h1>The framework value is: {}</h1>'''.format(language, framework)

    return '''<form method="POST">
                  Language: <input type="text" name="language"><br>
                  Framework: <input type="text" name="framework"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''


@app.route('/json-example', methods=["POST"])
def json_example():

    req_data = request.get_json()

    language = req_data['language']
    framework = req_data['framework']
    python_version = req_data['version_info']['python']
    example = req_data['example'][0]
    boolean_test = req_data['boolean_test']

    return '''
            The language value is: {}
            The framework value is: {}
            The Python version is: {}
            The item at index 0 in the example list is: {}
            The boolean value is: {}'''.format(language, framework, python_version, example, boolean_test)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
