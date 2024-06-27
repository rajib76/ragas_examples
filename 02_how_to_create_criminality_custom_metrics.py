import os

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate

from custom_metrics.criminality import criminality

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

data_samples = {
    'question': ['What do you want to do?'],
    'answer': ['I want to go to the park for a stroll']
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[criminality])
print(score)