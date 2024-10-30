import os
import openai
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY = openai.api_key = "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"
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain import RagasEvaluatorChain