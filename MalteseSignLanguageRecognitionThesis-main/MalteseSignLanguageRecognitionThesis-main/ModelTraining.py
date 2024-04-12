import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MalteseSignLanguageRecognitionThesis-main', 'largedata')
actions = np.array(['Kont', 'Dar', 'Flus', 'Missier', 'Passaport'])

# Define the sequence length and number of videos per action
sequence_length = 30  # Frames per sequence
no_sequences = 30     # Number of sequences per action

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in os.listdir(os.path.join(DATA_PATH, action)):
        window = []
        for frame_num in range(sequence_length):
            res_path = os.path.join(DATA_PATH, action, sequence, f"{frame_num}.npy")
            if os.path.exists(res_path):
                res = np.load(res_path)
                window.append(res)
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 1662)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=50, callbacks=[TensorBoard(log_dir='Logs')])

yhat = np.argmax(model.predict(X_test), axis=1)
ytrue = np.argmax(y_test, axis=1)
print(classification_report(ytrue, yhat, target_names=actions, zero_division=0))
print("Confusion Matrix")
print(multilabel_confusion_matrix(ytrue, yhat))
print("Accuracy")
print(accuracy_score(ytrue, yhat))

model.save('model1.h5')
