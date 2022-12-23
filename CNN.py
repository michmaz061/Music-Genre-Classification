import tensorflow as tf
import numpy as np
import sklearn
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import os
pxlSizex = 432
pxlSizey = 288
def load_img(indir):
    samples = []
    labels = []
    print('os.listdir(indir)= ', os.listdir(indir))  # RB test
    print('indir= ', indir)

    for class_dir in os.listdir(indir):
        if not os.path.isdir(indir + '/' + class_dir):
            continue

        print("Loading:", class_dir)

        for file in os.listdir(indir + '/' + class_dir):
            image = cv2.imread("{}/{}/{}".format(indir, class_dir, file))
            samples.append(image)  # dodanie obrazków
            labels.append(class_dir)  # dodanie klas

    samples = np.array(samples)
    labels = np.array(labels)
    # print('obrazek: ',image)
    print('obrazek.shape: ', image.shape)
    return samples, labels


# samples, labels = load_img('dataset1')
samples, labels = load_img('E:/PycharmProjects/Music-Genre-Classification/Data/images_original')
print('loaded', len(samples), ' samples')
print('classes', set(labels))


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same",input_shape=(pxlSizey,pxlSizex,3)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.25))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.Dense(10))
# model.add(tf.keras.layers.Activation("softmax"))
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(pxlSizey, pxlSizex,3)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# optimizer = tf.keras.optimizers.RMSprop(0.0001, decay = 1e-6)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
encoder = sklearn.preprocessing.LabelEncoder()
labels = encoder.fit_transform(labels)
labels = labels.astype(float)
EPOCHS = 100
skf = StratifiedKFold(n_splits=90, shuffle=True)
for train_idx, val_idx in skf.split(samples, labels):
    trainSamples = np.array(samples)[train_idx]
    trainLabels = np.array(labels)[train_idx]

    testSamples = np.array(samples)[val_idx]
    testLabels = np.array(labels)[val_idx]
    H = model.fit(trainSamples, trainLabels, epochs=EPOCHS, batch_size=10, validation_data=(testSamples, testLabels))
    testResults = model.predict(testSamples)
    print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
    print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
    print("Accuracy: ", accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1)))

