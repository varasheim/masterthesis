#imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


#load files. Replace filenames for running
evaluation_x = pd.read_pickle('evaluation-x-file')

evaluation_y = pd.read_pickle('evaluation-y-file')

X_training = pd.read_pickle('x-training-file')

y_training = pd.read_pickle('y-training-file')


#create dataframes for further processing
df_x_train = pd.DataFrame(X_training)
df_y_train = pd.DataFrame(y_training)
df_evaluation_x = pd.DataFrame(evaluation_x)
df_evaluation_y = pd.DataFrame(evaluation_y)

#change type of labels for use as input in model
df_y_train = df_y_train.astype(int)
df_evaluation_y = df_evaluation_y.astype(int)

#labels 
labels_train = df_y_train.values
labels_eval = df_evaluation_y.values

#create lists of labels
train_labels= []
for element in labels_train:
  train_labels.append(element[0])

test_labels= []
for element in labels_eval:
  test_labels.append(element[0])


#tokeize training data
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
training_encodings = tokenizer(df_x_train[0].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(df_evaluation_x[0].tolist(), truncation=True, padding=True, max_length=512)

#creating dataset objects
class RedditDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#create dataset objects following the method above 
train_dataset = RedditDataset(training_encodings, train_labels)
test_dataset = RedditDataset(test_encodings, test_labels)

#instantiate the XLNet model loaded from pre-trained model
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)

#instantiate trainer with model, arguments and training and evaluation datasets
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
)

#initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

#training of the model
trainer.train()


eval_results = trainer.evaluate()
test_results = trainer.evaluate(test_dataset)
print(test_results)


predictions = trainer.predict(test_dataset)
y_true = test_dataset.labels  
y_pred = np.argmax(predictions.predictions, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Accuracy: {accuracy:.2f}')

#create confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:", cm)
disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_true))
disp.plot(cmap='Blues', values_format='d')
plt.show()