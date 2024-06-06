#import
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#load files from ADHD Reddit Dataset, replace filename. 
evaluation_x = pd.read_pickle('eval-x-file')

evaluation_y = pd.read_pickle('eval-y-file')

X_training = pd.read_pickle('x-train-file')

y_training = pd.read_pickle('y-train-file')



#need to reshape the shape because keras requires three dimensions 
X_training = np.expand_dims(X_training, axis=2)
evaluation_x = np.expand_dims(evaluation_x, axis=2)

X_training = np.expand_dims(X_training, axis=3)
evaluation_x = np.expand_dims(evaluation_x, axis=3)

X_training = np.repeat(X_training, 1000, axis=2)
evaluation_x = np.repeat(evaluation_x, 1000, axis=2)


#build layers in CNN model
model = Sequential([
    Input(shape=(1000, 1000, 1)),
    Conv2D(8, 3), 
    MaxPooling2D(pool_size=2),
    Conv2D(8,3),
    MaxPooling2D(pool_size=2),
    Dense(64, activation='relu'),
    Flatten(),
    Dropout(0.5),
    Dense(3, activation='softmax'),
])

#compile model
model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy', metrics=['accuracy'])


#fit model and decide number of epochs
model.fit(
  X_training,
  to_categorical(y_training),
  epochs=2,
  validation_data=(evaluation_x, to_categorical(evaluation_y)),
)

#evaluation of the predictions
predictions = model.predict(evaluation_x)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(to_categorical(evaluation_y), axis=1)

accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

#create confusion matrices
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(true_labels))
disp.plot(cmap='Blues', values_format='d')
plt.show()