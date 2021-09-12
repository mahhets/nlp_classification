from dotenv import load_dotenv
from nlp_classification.web_server import app
import os

load_dotenv()

if __name__ == "__main__":
    app.run(
        host=os.getenv('SERVER_HOST'),
        port=os.getenv('SERVER_PORT')
    )