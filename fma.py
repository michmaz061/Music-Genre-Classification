import ast
import os
import librosa
import pandas as pd
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

root_path = os.path.join(os.path.abspath(__file__), os.path.pardir)


def spectrogram(music_path, track_id, genre, X, Y):
    remove_stereo = "sox '{}' 'tmp/{:06d}.mp3' remix 1-2".format(
        music_path, track_id)
    process = Popen(remove_stereo, shell=True, stdin=PIPE,
                    stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=root_path)
    output, errors = process.communicate()
    if errors:
        print(errors)
    # Create spectrogram
    if not os.path.exists(os.path.join(root_path, "spectrograms", genre)):
        os.makedirs(os.path.join(root_path, "spectrograms", genre))
    generate_spectrogram = "sox 'tmp/{:06d}.mp3' -n spectrogram -Y {} -X {} -m -r -o \
                                'spectrograms/{}/{:06d}.png'".format(
        track_id,
        Y,
        X,
        genre,
        track_id,
    )

    process = Popen(generate_spectrogram, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                    close_fds=True, cwd=root_path)
    output, errors = process.communicate()
    if errors:
        print(errors)


def get_all_music(path, labels, X, Y):
    print("Getting music files and converting it into spectrogram ...")
    music_files = librosa.util.find_files(directory=path, recurse=True)
    t = tqdm(music_files, desc='Bar desc', leave=True)
    if not os.path.exists(os.path.join(root_path, "tmp")):
        os.makedirs(os.path.join(root_path, "tmp"))
    for audio in t:
        t.set_description("file: {}".format(os.path.basename(audio)))
        track_id = os.path.splitext(os.path.basename(audio))[0].lstrip('0')
        genre = labels.loc[labels['track_id'] == int(track_id)]['genre'].values[0]
        spectrogram(audio, int(track_id), genre, X, Y)
        t.refresh()
    print("Spectrogram created")
    shutil.rmtree(os.path.join(root_path, "tmp"))


def slice_spectrograms(path, desired_size):
    spectrograms_path = os.path.join(root_path, path)
    print("Slicing All Spectrograms")
    t = tqdm(os.listdir(spectrograms_path), desc='Bar desc', leave=True)
    for genre_folder in t:
        for spectrogram in os.listdir(os.path.join(spectrograms_path, genre_folder)):
            if spectrogram.endswith('png'):
                t.set_description(
                    "file: {}/{}".format(genre_folder, spectrogram))
                slice_(os.path.join(root_path, path,
                                    genre_folder, spectrogram), desired_size)
                t.refresh()
    print("Spectrogram slice created")


def slice_(img_path, desired_size):
    img = Image.open(img_path)
    width, height = img.size
    samples_size = int(width / desired_size)
    genre = img_path.split('/')[-2]
    track_id = img_path.split('/')[-1]
    slice_path = 'preprocessing/slices/{}'.format(genre)

    if not os.path.exists(os.path.join(root_path, slice_path)):
        os.makedirs(os.path.join(root_path, slice_path))

    for i in range(samples_size):
        start_pixel = i * desired_size
        img_tmp = img.crop((start_pixel, 1, start_pixel +
                            desired_size, desired_size + 1))
        save_path = os.path.join(root_path, slice_path)
        img_tmp.save(save_path + "/{}_{}.png".format(track_id.rstrip('.png'), i))


def load_spectrogram(path, image_size):
    spectrogram = Image.open(path)
    spectrogram = spectrogram.resize(
        (image_size, image_size), resample=Image.ANTIALIAS)
    spectrogram_data = np.asarray(
        spectrogram, dtype=np.uint8).reshape(image_size, image_size, 1)
    return spectrogram_data / 255


def create_dataset_from_slices(spectrograms_per_genre, genres, slice_size, validation_ratio, test_ratio):
    data = []
    for genre in genres:
        print("-> Adding {}...".format(genre))
        filenames = os.listdir(os.path.join(root_path, "preprocessing", "slices", genre))
        filenames = [
            filename for filename in filenames if filename.endswith('.png')]
        filenames = filenames[:spectrograms_per_genre]
        shuffle(filenames)

        # Add data (X,y)
        for filename in filenames:
            img_data = load_spectrogram(os.path.join(
                root_path, "preprocessing", "slices", genre, filename), slice_size)
            label = [1. if genre == g else 0. for g in genres]
            data.append((img_data, label))

    # Shuffle data
    shuffle(data)

    # Extract X and y
    x, y = zip(*data)

    # Split data
    validation_len = int(len(x) * validation_ratio)
    test_len = int(len(x) * test_ratio)
    train_len = len(x) - (validation_len + test_len)

    # Prepare for Tflearn at the same time
    train_x = np.array(x[:train_len]).reshape([-1, slice_size, slice_size, 1])
    train_y = np.array(y[:train_len])
    validation_x = np.array(
        x[train_len:train_len + validation_len]).reshape([-1, slice_size, slice_size, 1])
    validation_y = np.array(y[train_len:train_len + validation_len])
    test_x = np.array(x[-test_len:]).reshape([-1, slice_size, slice_size, 1])
    test_y = np.array(y[-test_len:])
    return train_x, train_y, validation_x, validation_y, test_x, test_y


def model_assess(model, train_x, train_y, validation_x, validation_y, batch_size, epoch, title="Default"):
    model.fit(train_x, train_y, n_epoch=epoch, batch_size=batch_size, shuffle=True,
              validation_set=(validation_x, validation_y), snapshot_step=100)
    pred = model.predict(test_x)
    print('Accuracy', title, ':', round(accuracy_score(test_y, pred), 5), '\n')


# file_path = 'F:/fma_metadata/tracks.csv'
# subset = 'medium'
# tracks = pd.read_csv(file_path, index_col=0, header=[0, 1])
#
# COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
#            ('track', 'genres'), ('track', 'genres_all')]
# for column in COLUMNS:
#     tracks[column] = tracks[column].map(ast.literal_eval)
#
# COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
#            ('album', 'date_created'), ('album', 'date_released'),
#            ('artist', 'date_created'), ('artist', 'active_year_begin'),
#            ('artist', 'active_year_end')]
# for column in COLUMNS:
#     tracks[column] = pd.to_datetime(tracks[column])
#
# SUBSETS = ('small', 'medium', 'large')
# try:
#     tracks['set', 'subset'] = tracks['set', 'subset'].astype(
#         'category', categories=SUBSETS, ordered=True)
# except (ValueError, TypeError):
#     tracks['set', 'subset'] = tracks['set', 'subset'].astype(
#         pd.CategoricalDtype(categories=SUBSETS, ordered=True))
#
# COLUMNS = [('track', 'genre_top'), ('track', 'license'),
#            ('album', 'type'), ('album', 'information'),
#            ('artist', 'bio')]
# for column in COLUMNS:
#     tracks[column] = tracks[column].astype('category')
#
# subset = tracks.index[tracks['set', 'subset'] <= subset]
# labels = tracks.loc[subset, ('track', 'genre_top')]
# labels.name = 'genre'
# labels = labels.dropna()
# labels.to_csv('preprocessing/train_labels.csv', header=True)
labels = pd.read_csv('preprocessing/train_labels.csv')
# print(labels)
dataset_path = 'F:/fma_medium'
get_all_music(dataset_path, labels, 20, 200)
slice_spectrograms("spectrograms", 198)
# genres = labels['genre'].unique()
# train_x, train_y, validation_x, validation_y, test_x, test_y = create_dataset_from_slices(2000, genres, 198, 0.15, 0.1)
#
# nb = GaussianNB()
# model_assess(nb, train_x, train_y, validation_x, validation_y, 128, 20, "Naive Bayes")
#
# sgd = SGDClassifier(max_iter=5000, random_state=0)
# model_assess(sgd, train_x, train_y, validation_x, validation_y, 128, 20, "Stochastic Gradient Descent")
#
# knn = KNeighborsClassifier(n_neighbors=64)
# model_assess(knn, train_x, train_y, validation_x, validation_y, 128, 20, "KNN")
#
# tree = DecisionTreeClassifier()
# model_assess(tree, train_x, train_y, validation_x, validation_y, 128, 20, "Decission trees")
#
# randforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
# model_assess(randforest, train_x, train_y, validation_x, validation_y, 128, 20, "Random Forest")
#
# xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
# model_assess(xgb, train_x, train_y, validation_x, validation_y, 128, 20, "Cross Gradient Booster")
