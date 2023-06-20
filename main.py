from typing import Annotated
from fastapi import FastAPI, Form
from transformers import AutoTokenizer
from transformers import BertForQuestionAnswering
from transformers import pipeline

app = FastAPI()

model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')

tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)


def question_answerer(question,context):
    nlp({
    'question':question,
    'context':context
})


@app.post("/qa/")
async def qa(context: Annotated[str, Form()], question: Annotated[str, Form()]):
    qamodel=question_answerer(question=question, context=context)
    return {"answer": qamodel[0]["answer"]}

