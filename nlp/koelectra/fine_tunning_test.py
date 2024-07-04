from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from time import time

start_time = time()

model_path = '../output/koelectra'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

test_file_path = './train_data/test_1000.txt'
with open(test_file_path, 'r', encoding='utf8') as file:
    test_lines = file.readlines()

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).item()
    return predictions

cnt = 0
for line in test_lines:
    label, text = line.strip().split(',', 1)
    prediction = predict(text, model, tokenizer)
    if prediction == 1:
        predicted_label = '대상'
        with open('./log.txt', 'a') as file:
            formmated_string = '대상,{}\n'.format(text)
            file.write(formmated_string)
    else:
        predicted_label = '비대상'
        with open('./log.txt', 'a') as file:
            formmated_string = '비대상,{}\n'.format(text)
            file.write(formmated_string)

end_time = time()
print('Elpased time : ', end_time - start_time)
