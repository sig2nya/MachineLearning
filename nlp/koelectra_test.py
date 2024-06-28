from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

file_path = './train_data/train_500000'

with open(file_path, 'r', encoding='utf8') as file:
    lines = file.readlines()

data = [line.strip().split(',') for line in lines]
data = [row for row in data if len(row) == 2]

df = pd.DataFrame(data, columns=['label', 'text'])
df['label'] = df['label'].apply(lambda x : 1 if x == '대상' else 0)
print(df[:10])

dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

split_datasets = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets['train']
test_dataset = split_datasets['test']

model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=2)

training_args = TrainingArguments(
        output_dir='.results',
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
)

trainer.train()

save_directory = '../output/koelectra'
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print('모델 저장 완료')
