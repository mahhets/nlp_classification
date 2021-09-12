import pandas as pd
import pickle
import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask import request
from environment_reference import EnvironmentReference

load_dotenv()

model_path = os.getenv(EnvironmentReference.SAVE_MODEL_PATH)
print(model_path)

with open(f'{model_path}MNB_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(model)

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
async def predict():
    text = str(request.args.get('text', ''))
    print(text)
    try:
        if not text:
            raise KeyError('Parameter "text" is required')
        data = {
            'text': text
        }

        temp_df = pd.DataFrame(data)
        print(temp_df)

    except (KeyError, ValueError) as err:
        return jsonify({"error": str(err)}), 400
