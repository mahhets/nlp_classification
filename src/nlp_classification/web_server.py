import pandas as pd
import pickle
import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask import request
from environment_reference import EnvironmentReference

load_dotenv()

model_path = os.getenv(EnvironmentReference.SAVE_MODEL_PATH)

with open(f'{model_path}MNB_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def general():
    welcome = """REST API на основе ML-классификатора.\n
    
    Классифицирует тектовые сообщения по 2-м тематикам: Python / DataScience.\n\n
    
    Пример запроса: "server_adress/predict?text=В питоне очень приятный синтаксис"
    """
    return welcome


@app.route('/predict', methods=['GET'])
def predict():
    text = str(request.args.get('text', ''))
    try:
        if not text:
            raise KeyError('Parameter "text" is required')
        data = {
            'text': [text]
        }

        temp_df = pd.DataFrame(data)
        _predict = model.predict_proba(temp_df)[0][1]
        # Найденный порог для классификации
        if _predict < 0.442:
            return jsonify({"Chat": 'Python'}), 200
        else:
            return jsonify({"Chat": 'DS'}), 200

    except (KeyError, ValueError) as err:
        return jsonify({"error": str(err)}), 400


if __name__ == "__main__":
    app.run(
        host=os.getenv('SERVER_HOST'),
        port=os.getenv('SERVER_PORT')
    )
