import sys
import tensorflow as tf
import numpy as np
import sklearn
import cv2
import gc
from sklearn.model_selection import KFold
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
            samples.append(image)
            labels.append(class_dir)

    samples = np.array(samples)
    labels = np.array(labels)
    # print('obrazek: ',image)
    print('obrazek.shape: ', image.shape)
    return samples, labels


file_path = 'outputcnn1.txt'
sys.stdout = open(file_path, "w")
# samples, labels = load_img('dataset1')
samples, labels = load_img('E:/PycharmProjects/Music-Genre-Classification/Data/images_original')
print('loaded', len(samples), ' samples')
print('classes', set(labels))
gc.enable()
print("---CNN---")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same",input_shape=(pxlSizey,pxlSizex,3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation("softmax"))

optimizer = tf.keras.optimizers.RMSprop(0.0001, decay = 1e-6)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = labels.astype(float)
EPOCHS = 100

kf = KFold(n_splits=10, shuffle=True)
for train_idx, val_idx in kf.split(samples, labels):
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

input_shapem=(pxlSizey,pxlSizex,3)
print("---CRNN---")
model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shapem))
model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
model2.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model2.add(tf.keras.layers.Dropout(0.2))
model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
model2.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(tf.keras.layers.Dropout(0.2))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
model2.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(tf.keras.layers.Dropout(0.1))
model2.add(tf.keras.layers.BatchNormalization())
model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
model2.add(tf.keras.layers.Reshape((50, 512)))
model2.add(tf.keras.layers.LayerNormalization())
model2.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)))
model2.add(tf.keras.layers.Dropout(0.1))
model2.add(tf.keras.layers.LayerNormalization())
model2.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256)))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(256, activation='relu'))
model2.add(tf.keras.layers.Dense(10, activation='softmax'))


optimizer2 = tf.keras.optimizers.RMSprop(0.0001, decay = 1e-6)
model2.compile(loss='categorical_crossentropy', optimizer=optimizer2,metrics=['accuracy'])

kf2 = KFold(n_splits=10, shuffle=True)
for train_idx, val_idx in kf2.split(samples, labels):
    trainSamples = np.array(samples)[train_idx]
    trainLabels = np.array(labels)[train_idx]

    testSamples = np.array(samples)[val_idx]
    testLabels = np.array(labels)[val_idx]
    H = model2.fit(trainSamples, trainLabels, epochs=200, validation_data=(testSamples,testLabels))
    testResults = model2.predict(testSamples)
    print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
    print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
    print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
    print("Accuracy: ", accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1)))


