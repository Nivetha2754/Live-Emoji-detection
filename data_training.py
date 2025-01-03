import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Ensure compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize variables
is_init = False
label = []
dictionary = {}
c = 0

# Load .npy files
for i in os.listdir():
    if i.endswith(".npy"):  # Check for .npy files
        data = np.load(i)
        if not is_init:
            is_init = True
            X = data
            size = data.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1


if not is_init:
    raise ValueError("No .npy files found in the current directory.")

for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]


input_shape = (X.shape[1],)  # Ensure shape is a tuple
ip = Input(shape=input_shape)
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])


model.fit(X, y, epochs=50)


model.save("model.h5")
np.save("labels.npy", np.array(label))

