from flask import Flask, request, jsonify
from flask import send_from_directory
from flask import Response,stream_with_context
from predict import model_pred
app = Flask(__name__)
# CORS(app)

app.config["files"] =  './input/'
app.config["output_path"] = './output/'


@app.route('/extract-senti', methods=['POST'])
def extract_ner_re():
      data = request.get_data()
      try:
          data = data.decode('utf-8', errors='replace')
      except UnicodeDecodeError:
          data = data.decode('latin-1', errors='replace')
      res_score, result = model_pred(data)
      return {'score':str(res_score),'Result':result}


def runflask():
    app.run(host="0.0.0.0", port=6234, debug=False)


if __name__ == '__main__':
    runflask()