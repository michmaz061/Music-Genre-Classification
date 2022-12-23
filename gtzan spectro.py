import ast
import bz2
import os
import pickle
import classic as cl
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import gc
import sys
from pandas.api.types import CategoricalDtype
from librosa.display import specshow
from librosa.feature import melspectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import shutil
import numpy as np
from random import shuffle
from PIL import Image
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT

file_output_img = 'E:/PycharmProjects/Music-Genre-Classification/gtzan/spectrograms/images/'
file_output = 'E:/PycharmProjects/Music-Genre-Classification/gtzan/spectrograms/'
file_output_slic = 'E:/PycharmProjects/Music-Genre-Classification/gtzan/spectrograms/slices/'
file_output_slic2 = 'E:/PycharmProjects/Music-Genre-Classification/gtzan/spectrograms/images/'
file_output_slic3 = 'E:/PycharmProjects/Music-Genre-Classification/Data/images_original/'
audio_location = "E:/PycharmProjects/Music-Genre-Classification/Data/genres_original/"
i = 0


def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, mono=True, duration=conf.duration)
    # trim silence
    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples:  # long enough
        if trim_long_data:
            y = y[0:0 + conf.samples]
    else:  # pad blank
        padding = conf.samples - len(y)  # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y


class conf:
    # Preprocessing settings
    sampling_rate = 22050
    duration = 5
    hop_length = 512
    fmin = 20
    fmax = 8000
    n_mels = 128
    n_fft = 2048
    samples = sampling_rate * duration



def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=conf.sampling_rate, n_fft=conf.n_fft,
                                                 hop_length=conf.hop_length, n_mels=conf.n_mels, fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram



def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    return mels


def load_spectrogram(path, image_size):
    spectrogram = Image.open(path)
    spectrogram = spectrogram.resize((image_size, image_size), resample=Image.LANCZOS)
    spectrogram_data = np.asarray(spectrogram, dtype=np.uint8).reshape(-1)
    return spectrogram_data/255


def create_dataset_from_slices(slice_size):
    data = []
    # for every directory in the images folder
    for direct in os.listdir(file_output_slic2):
        print("-> Adding {}...".format(direct))
        filenames = os.listdir(os.path.join(file_output_slic2, direct))
        filenames = [filename for filename in filenames if filename.endswith('.png')]
        filenames = filenames
        shuffle(filenames)
        for filename in filenames:
            img_data = load_spectrogram(os.path.join(file_output_slic2, direct, filename), slice_size)
            id = filename.split('_')[0]
            data.append([img_data, direct])
    # Shuffle data
    shuffle(data)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # Extract X and y
    x, y = zip(*data)
    del data, img_data,
    for train_idx, val_idx in skf.split(x, y):
        # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        X_tr = np.array(x)[train_idx]
        y_tr = np.array(y)[train_idx]

        X_val = np.array(x)[val_idx]
        y_val = np.array(y)[val_idx]
        cl.main(X_tr, y_tr, X_val, y_val)


# def model_assess(model, train_x, train_y, title="Default"):
#     model.fit(train_x, train_y)
#     pred = model.predict(test_x)
#     print('Accuracy', title, ':', round(accuracy_score(test_y, pred), 5), '\n')


def rename_file(img_name):
    img_name = img_name.split("/")
    img_name = img_name[-1]
    img_name = img_name.split('.wav')[0]
    img_name += ".jpg"
    return img_name


def save_image_from_sound(tid,foldername):
    filepath = os.path.join(audio_location,foldername,tid)
    filename = rename_file(filepath)
    x = read_as_melspectrogram(conf, filepath, trim_long_data=False, debug_display=True)
    # x_color = mono_to_color(x)

    plt.imshow(x, interpolation='nearest')
    plt.axis('off')
    adr = file_output_img + filename[:3]
    if not os.path.exists(file_output_img + foldername):
        os.makedirs(file_output_img + foldername)
    final_file = file_output_img + filename
    plt.savefig(final_file, bbox_inches='tight', pad_inches=0)
    print('saving: ', final_file)
    plt.close()
    del x
    gc.collect()


def slice_spectrograms(desired_size):
    spectrograms_path = file_output_img
    print("Slicing All Spectrograms")
    t = tqdm(os.listdir(spectrograms_path), desc='Bar desc', leave=True)
    for folder in t:
        for spectrogram in os.listdir(os.path.join(spectrograms_path, folder)):
            if spectrogram.endswith('jpg'):
                t.set_description(
                    "file: {}/{}".format(folder, spectrogram))
                slice_(os.path.join(file_output_img, folder, spectrogram), desired_size)
                t.refresh()
    print("Spectrogram slice created")


def slice_(img_path, desired_size):
    img = Image.open(img_path)
    width, height = img.size
    samples_size = int(width / desired_size)
    track_id = img_path.split('/')[-1]
    slice_path = 'slices/{}'.format(track_id.split("\\")[0])
    adr = os.path.join(file_output_slic, slice_path)
    if not os.path.exists(os.path.join(file_output, slice_path)):
        os.makedirs(os.path.join(file_output, slice_path))

    for i in range(samples_size):
        start_pixel = i * desired_size
        img_tmp = img.crop((start_pixel, 1, start_pixel +
                            desired_size, desired_size + 1))
        save_path = os.path.join(file_output, "slices")
        adr2 = save_path + "/{}_{}.jpg".format(track_id.rstrip('.jpg'), i)
        img_tmp.save(save_path + "/{}_{}.jpg".format(track_id.rstrip('.jpg'), i))


# for folder in os.listdir(audio_location):
#     i += 1
#     if i == 11:
#         break
#     for file in os.listdir(audio_location + "/" + folder):
#         try:
#             save_image_from_sound(file,folder)
#         except Exception as e:
#             print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)

# slice_spectrograms(198)
file_path = 'outputclass2.txt'
sys.stdout = open(file_path, "w")
gc.enable()
create_dataset_from_slices(198)


# nb = GaussianNB()
# model_assess(nb, train_x, train_y, "Naive Bayes")

# sgd = SGDClassifier(max_iter=1000, random_state=0)
# model_assess(sgd, train_x, train_y, "Stochastic Gradient Descent")
#
# knn = KNeighborsClassifier(n_neighbors=10)
# model_assess(knn, train_x, train_y, "KNN")

# tree = DecisionTreeClassifier()
# model_assess(tree, train_x, train_y, "Decission trees")
#
# randforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
# model_assess(randforest, train_x, train_y, "Random Forest")
#
# xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
# model_assess(xgb, train_x, train_y, "Cross Gradient Booster")

# hgb = HistGradientBoostingClassifier(learning_rate=0.05)
# model_assess(hgb, train_x, train_y, "Hist Gradient Booster")