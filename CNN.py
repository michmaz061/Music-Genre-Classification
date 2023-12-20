import sys
import tensorflow as tf
import numpy as np
import sklearn
import cv2
import gc
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import os

pxlSizex = 432
pxlSizey = 288
accumulation_steps = 4

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
            image = cv2.resize(image, (pxlSizex, pxlSizey))
            samples.append(image)
            labels.append(class_dir)

    samples = np.array(samples)
    labels = np.array(labels)
    # print('obrazek: ',image)
    print('obrazek.shape: ', image.shape)
    return samples, labels

print(tf.config.list_physical_devices('GPU'))
file_path = 'outputcrnn11_mel_images15-30_512,0,11025,256,2048.txt'
sys.stdout = open(file_path, "w")
# samples, labels = load_img('dataset1')
#spectro
# samples, labels = load_img("/mnt/e/PycharmProjects/Music-Genre-Classification/data/images_original")
#mel
samples, labels = load_img("/mnt/e/PycharmProjects/Music-Genre-Classification/gtzan/spectrograms/images15-30_512,0,11025,256,2048")
print('loaded', len(samples), ' samples')
print('classes', set(labels))
gc.enable()
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = labels.astype(float)
EPOCHS = 100
input_shapem = (pxlSizey, pxlSizex, 3)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
ss = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
# print("---CNN---")
# with tf.device('/gpu:0'):
#     for train_idx, val_idx in sss.split(samples, labels):
#         # (trainSamples, testSamples, trainLabels, testLabels) = sklearn.model_selection.train_test_split(samples, labels,random_state=33)
#         model = tf.keras.models.Sequential()
#         model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", input_shape=(pxlSizey, pxlSizex, 3)))
#         model.add(tf.keras.layers.BatchNormalization())
#         model.add(tf.keras.layers.Activation("relu"))
#         model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#         model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
#         model.add(tf.keras.layers.BatchNormalization())
#         model.add(tf.keras.layers.Activation("relu"))
#         model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#         model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
#         model.add(tf.keras.layers.BatchNormalization())
#         model.add(tf.keras.layers.Activation("relu"))
#         model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#         model.add(tf.keras.layers.Dropout(0.25))
#         model.add(tf.keras.layers.Flatten())
#         model.add(tf.keras.layers.Dense(512))
#         model.add(tf.keras.layers.Activation("relu"))
#         model.add(tf.keras.layers.Dense(10))
#         model.add(tf.keras.layers.Activation("softmax"))
#         optimizer = tf.keras.optimizers.Adam(0.0001)
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#         trainSamples = np.array(samples)[train_idx]
#         trainLabels = np.array(labels)[train_idx]
#         testSamples = np.array(samples)[val_idx]
#         testLabels = np.array(labels)[val_idx]
#         H = model.fit(trainSamples, trainLabels, epochs=EPOCHS, batch_size=10, validation_data=(testSamples, testLabels))
#         testResults = model.predict(testSamples)
#         print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
#         print("Accuracy: ", accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         gc.collect()


# print("---CRNN---")
# with tf.device('/gpu:0'):
#     for train_idx, val_idx in sss.split(samples, labels):
#         trainSamples = np.array(samples)[train_idx]
#         trainLabels = np.array(labels)[train_idx]
#
#         testSamples = np.array(samples)[val_idx]
#         testLabels = np.array(labels)[val_idx]
#
#         model2 = tf.keras.models.Sequential()
#         model2.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shapem))
#         model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model2.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#         model2.add(tf.keras.layers.Dropout(0.2))
#         model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model2.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#         model2.add(tf.keras.layers.Dropout(0.2))
#         model2.add(tf.keras.layers.BatchNormalization())
#         model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model2.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#         model2.add(tf.keras.layers.Dropout(0.1))
#         model2.add(tf.keras.layers.BatchNormalization())
#         model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model2.add(tf.keras.layers.Reshape((50, 512)))
#         model2.add(tf.keras.layers.LayerNormalization())
#         model2.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)))
#         model2.add(tf.keras.layers.Dropout(0.1))
#         model2.add(tf.keras.layers.LayerNormalization())
#         model2.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256)))
#         model2.add(tf.keras.layers.Flatten())
#         model2.add(tf.keras.layers.Dense(256, activation='relu'))
#         model2.add(tf.keras.layers.Dense(10, activation='softmax'))
#
#         optimizer2 = tf.keras.optimizers.RMSprop(0.0001)
#         model2.compile(loss='categorical_crossentropy', optimizer=optimizer2, metrics=['accuracy'])
#         H = model2.fit(trainSamples, trainLabels, epochs=200, validation_data=(testSamples, testLabels))
#         testResults = model2.predict(testSamples)
#         print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
#         print("Accuracy: ", accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         gc.collect()

print("---CRNN11  vRB2---")
with tf.device('/gpu:0'):
    for train_idx, val_idx in sss.split(samples, labels):
        trainSamples = np.array(samples)[train_idx]
        trainLabels = np.array(labels)[train_idx]

        testSamples = np.array(samples)[val_idx]
        testLabels = np.array(labels)[val_idx]

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=input_shapem))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Reshape((16 * 25, 64)))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        H = model.fit(trainSamples, trainLabels, epochs=300, batch_size=10, validation_data=(testSamples, testLabels))
        testResults = model.predict(testSamples)
        print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
        print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
        print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
        print("Accuracy: ", accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
        gc.collect()


# print("---CRNN11  vRB6---")
# with tf.device('/gpu:0'):
#     for train_idx, val_idx in sss.split(samples, labels):
#         trainSamples = np.array(samples)[train_idx]
#         trainLabels = np.array(labels)[train_idx]
#
#         testSamples = np.array(samples)[val_idx]
#         testLabels = np.array(labels)[val_idx]
#
#         model = tf.keras.models.Sequential()
#         model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu',input_shape=input_shapem))
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#         model.add(tf.keras.layers.Dropout(0.2))
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#         model.add(tf.keras.layers.Dropout(0.2))
#         model.add(tf.keras.layers.BatchNormalization())
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#         model.add(tf.keras.layers.Dropout(0.1))
#         model.add(tf.keras.layers.BatchNormalization())
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Reshape((50, 512)))
#         model.add(tf.keras.layers.LayerNormalization())
#         model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)))
#         model.add(tf.keras.layers.Dropout(0.1))
#         model.add(tf.keras.layers.LayerNormalization())
#         model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256)))
#         model.add(tf.keras.layers.Flatten())
#         model.add(tf.keras.layers.Dense(256, activation='relu'))
#         model.add(tf.keras.layers.Dense(10, activation='softmax'))
#         optimizer = tf.keras.optimizers.RMSprop(0.0001, momentum = 1e-6)
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
#
#         H = model.fit(trainSamples, trainLabels, epochs=200, validation_data=(testSamples,testLabels))
#         testResults = model.predict(testSamples)
#         print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
#         print("Accuracy: ", accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         gc.collect()


# sys.stdout.close()
# file_path2 = 'outputcrnn11_mel_images15-30_512,20,8000,128,4096.txt'
# sys.stdout = open(file_path2, "w")
# print("---CRNN11  vRB2---")
# with tf.device('/gpu:0'):
#     for train_idx, val_idx in sss.split(samples2, labels):
#         trainSamples = np.array(samples2)[train_idx]
#         trainLabels = np.array(labels2)[train_idx]
#
#         testSamples = np.array(samples2)[val_idx]
#         testLabels = np.array(labels2)[val_idx]
#
#         model = tf.keras.models.Sequential()
#         model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=input_shapem))
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#         model.add(tf.keras.layers.Dropout(0.1))
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#         model.add(tf.keras.layers.Dropout(0.1))
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#         model.add(tf.keras.layers.Dropout(0.1))
#         model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#         model.add(tf.keras.layers.Dense(64))
#         model.add(tf.keras.layers.Reshape((16 * 25, 64)))
#         model.add(tf.keras.layers.LayerNormalization())
#         model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)))
#         model.add(tf.keras.layers.LayerNormalization())
#         model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)))
#         model.add(tf.keras.layers.Flatten())
#         model.add(tf.keras.layers.Dense(256, activation='relu'))
#         model.add(tf.keras.layers.Dense(10, activation='softmax'))
#         optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#         H = model.fit(trainSamples, trainLabels, epochs=300, batch_size=10, validation_data=(testSamples, testLabels))
#         testResults = model.predict(testSamples)
#         print(confusion_matrix(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print(classification_report(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         print("Cohen's Kappa: {}".format(cohen_kappa_score(testLabels.argmax(axis=1), testResults.argmax(axis=1))))
#         print("Accuracy: ", accuracy_score(testLabels.argmax(axis=1), testResults.argmax(axis=1)))
#         gc.collect()
#
