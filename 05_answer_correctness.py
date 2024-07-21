import os

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import AnswerCorrectness

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
data_samples = {
    'question': ['When was the first super bowl?'],
    'answer': ['The game of base ball was held on January 15 1967.'],
    'ground_truth': ['The first superbowl was held on January 15 1967.',]
}
dataset = Dataset.from_dict(data_samples)
openai_model = ChatOpenAI(
    model="gpt-4"
)
# answer_correctness = AnswerCorrectness(weights=[0.25,0.75])
score = evaluate(dataset, metrics=[AnswerCorrectness(weights=[0,1])],llm=openai_model)
print(score.to_pandas())
