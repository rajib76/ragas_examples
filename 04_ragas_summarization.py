import os

from datasets import Dataset
from dotenv import load_dotenv
from ragas.metrics import summarization_score
from ragas import evaluate
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
data_samples = {
    'contexts' : [["Rajib is a software engineer. He lives in San Jose. He is expert in python"]],
    'summary': ["rajib is a python skilled engineer"]
}
print("length of context ", len("Rajib is a software engineer. He lives in San Jose. He is expert in python"))
print("length of sumamry ", len("rajib is a python skilled engineer"))
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[summarization_score])
print(score)
print(score.to_pandas())

