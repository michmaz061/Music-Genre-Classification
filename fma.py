import ast
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import gc
from pandas.api.types import CategoricalDtype
from librosa.display import specshow
from librosa.feature import melspectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

file_output_img = 'E:/PycharmProjects/Music-Genre-Classification/spectrograms/images/'
file_output = 'E:/PycharmProjects/Music-Genre-Classification/spectrograms/'
file_output_slic = 'E:/PycharmProjects/Music-Genre-Classification/spectrograms/slices/'
audio_location = "F:/fma_medium/"


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


# Thanks to the librosa library, generating the mel-spectogram from the audio file

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=conf.sampling_rate, n_fft=conf.n_fft,
                                                 hop_length=conf.hop_length, n_mels=conf.n_mels, fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


# Adding both previous function together

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    return mels


def load_spectrogram(path, image_size):
    spectrogram = Image.open(path)
    spectrogram = spectrogram.resize((image_size, image_size), resample=Image.LANCZOS)
    spectrogram_data = np.asarray(spectrogram, dtype=np.uint8).reshape(-1, image_size, image_size, 1)
    return spectrogram_data


def create_dataset_from_slices(genres, slice_size, test_ratio):
    data = []
    # for every directory in the images folder
    for direct in os.listdir(file_output_slic):
        print("-> Adding {}...".format(direct))
        filenames = os.listdir(os.path.join(file_output_slic, direct))
        filenames = [filename for filename in filenames if filename.endswith('.jpg')]
        filenames = filenames
        shuffle(filenames)
        for filename in filenames:
            img_data = load_spectrogram(os.path.join(file_output_slic, direct, filename), slice_size)
            id = filename.split('_')[0]
            data.append([img_data, genres['genre'].where(genres['track_id'] == int(id.lstrip("0"))).dropna().values[0]])

    # Shuffle data
    shuffle(data)

    # Extract X and y
    x, y = zip(*data)

    test_len = int(len(x) * test_ratio)
    train_len = len(x) - test_len

    # Prepare for Tflearn at the same time
    train_x = np.array(x[:train_len])
    train_y = np.array(y[:train_len])
    test_x = np.array(x[-test_len:])
    test_y = np.array(y[-test_len:])
    return train_x, train_y, test_x, test_y


def model_assess(model, train_x, train_y, title="Default"):
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print('Accuracy', title, ':', round(accuracy_score(test_y, pred), 5), '\n')


def rename_file(img_name):
    img_name = img_name.split("/")
    img_name = img_name[-1]
    img_name = img_name.split('.')[0]
    img_name += ".jpg"
    return img_name


def save_image_from_sound(tid):
    filepath = get_audio_path(audio_location, tid)
    filename = rename_file(filepath)
    x = read_as_melspectrogram(conf, filepath, trim_long_data=False, debug_display=True)
    # x_color = mono_to_color(x)

    plt.imshow(x, interpolation='nearest')
    plt.axis('off')
    if not os.path.exists(file_output_img + filename[:3]):
        os.makedirs(file_output_img + filename[:3])
    final_file = file_output_img + filename
    plt.savefig(final_file, bbox_inches='tight', pad_inches=0)
    print('saving: ', final_file)
    plt.close()
    del x
    gc.collect()


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


# file_path = 'F:/fma_metadata/tracks.csv'
# subset = 'medium'
# tracks = pd.read_csv(file_path, index_col=0, header=[0, 1])
#

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
    slice_path = 'slices/{}'.format(track_id[:3])

    if not os.path.exists(os.path.join(file_output, slice_path)):
        os.makedirs(os.path.join(file_output, slice_path))

    for i in range(samples_size):
        start_pixel = i * desired_size
        img_tmp = img.crop((start_pixel, 1, start_pixel +
                            desired_size, desired_size + 1))
        save_path = os.path.join(file_output, "slices")
        img_tmp.save(save_path + "/{}_{}.jpg".format(track_id.rstrip('.jpg'), i))

def load(filepath):
    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


#
# subset = tracks.index[tracks['set', 'subset'] <= subset]
# labels = tracks.loc[subset, ('track', 'genre_top')]
# labels.name = 'genre'
# labels = labels.dropna()
# labels.to_csv('preprocessing/train_labels.csv', header=True)
def read_metadata(name):
    return load('F:/fma_metadata/{0}.csv'.format(name))


# genres = read_metadata('genres')
# features = read_metadata('features')
# echonest = read_metadata('echonest')
# tracks = read_metadata('tracks')
# medium_tracks = tracks[tracks[('set', 'subset')] <= 'medium']
# tracks_spectro = medium_tracks[[('track', 'genre_top'), ('track', 'duration')]]
genres = pd.read_csv('preprocessing/train_labels.csv')
# tids = tracks_spectro.index
# count = 0
# for tid in tids:
#     try:
#         save_image_from_sound(tid)
#         count += 1
#     except Exception as e:
#         print('{}: {}'.format(tid, repr(e)))
# slice_spectrograms(198)
train_x, train_y, test_x, test_y = create_dataset_from_slices(genres, 198, 0.3)

# nb = GaussianNB()
# model_assess(nb, train_x, train_y, "Naive Bayes")

sgd = SGDClassifier(max_iter=5000, random_state=0)
model_assess(sgd, train_x, train_y, "Stochastic Gradient Descent")

knn = KNeighborsClassifier(n_neighbors=64)
model_assess(knn, train_x, train_y, "KNN")

tree = DecisionTreeClassifier()
model_assess(tree, train_x, train_y, "Decission trees")

randforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess(randforest, train_x, train_y, "Random Forest")

xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess(xgb, train_x, train_y, "Cross Gradient Booster")
