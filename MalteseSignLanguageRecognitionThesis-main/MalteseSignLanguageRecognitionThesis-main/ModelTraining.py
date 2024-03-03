import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

print("Hello")

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('largedata') 

# Actions that we try to detect  actions = np.array(['Familja', 'Missier', 'Bieb'])
actions = np.array(['Account','Flus', 'Missier', 'Passport','ThumbsUp'])


# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0

label_map = {label:num for num, label in enumerate(actions)}

label_map # When this is added accuracy skyrockets IMP

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

np.array(sequences).shape
np.array(labels).shape
X = np.array(sequences)
X.shape
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_test.shape

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# Creating the Model
model = Sequential()

# CNN
#inp= tf.reshape(len(X), 1, X.shape[1])
#inp = tf.reshape(len(y_train), 1, y_train.shape[1])
#X_train= tf.reshape(len(X_train), 1, X_train.shape[1])

#model.add(TimeDistributed(Conv2D(64, kernel_size=3, activation='relu', input_shape=(30,1662))))
#model.add(TimeDistributed(Conv2D(32, kernel_size=3, activation='relu', input_shape=(30,1662))))
#model.add(TimeDistributed(Flatten()))

# LTSM
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


#ERRORS
#train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#valid_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))


model.fit(X_train, y_train, epochs=50, callbacks=[tb_callback])

model.summary()


res = model.predict(X_test)
actions[np.argmax(res[4])]
actions[np.argmax(y_test[4])]

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
report = multilabel_confusion_matrix(ytrue, yhat)
accuracy = accuracy_score(ytrue, yhat)

print(classification_report(ytrue, yhat,target_names=actions))
print("Confusion Matrix")
print(report)
print("Accuracy")
print(accuracy)

#predicted = model.predict(X_test)
#report = classification_report(y_test, predicted)
#print(report)
#y_test_arg = np.argmax(y_test,axis=1)
#Y_pred = np.argmax(model.predict(X_test),axis=1)

#res = model.predict(X_test)
#print(actions[np.argmax(res[4])])
#print(actions[np.argmax(y_test[4])])

model.save('model1.h5')

#print('Confusion Matrix') # FIX https://datascience.stackexchange.com/questions/93751/valueerror-classification-metrics-cant-handle-a-mix-of-multilabel-indicator-an
#print(confusion_matrix(y_test_arg, Y_pred))
#print (classification_report(y_test_arg, Y_pred))