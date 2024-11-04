import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

train = pd.read_csv("emnist_models\\emnist-byclass-train.csv", delimiter=',')
test = pd.read_csv("emnist_models\\emnist-byclass-test.csv", delimiter=',')
mapp = pd.read_csv("dictfile.txt", delimiter=' ')

print("Train: %s, Test: %s, Map: %s" % (train.shape, test.shape, mapp.shape))

# Constants
HEIGHT = 28
WIDTH = 28

# Split x and y
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
del train

test_x = test.iloc[:, 1:]
test_y = test.iloc[:, 0]
del test

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# Flip and rotate image
train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)
print("train_x:", train_x.shape)

test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)
print("test_x:", test_x.shape)

# Normalise
train_x = train_x.astype('float32')
train_x /= 255
test_x = test_x.astype('float32')
test_x /= 255

# Reshape image for RFC
train_x = train_x.reshape(-1, HEIGHT * WIDTH)
test_x = test_x.reshape(-1, HEIGHT * WIDTH)

# Split into train and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Initialize RFC
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(train_x, train_y)

# Predict on validation set
val_predictions = model.predict(val_x)
val_accuracy = accuracy_score(val_y, val_predictions)
print("Validation Accuracy:", val_accuracy)

# Predict on test set
test_predictions = model.predict(test_x)
test_accuracy = accuracy_score(test_y, test_predictions)
print("Test Accuracy:", test_accuracy)

# Save the model 
joblib.dump(model, 'emnist_rfc_model.sav')


