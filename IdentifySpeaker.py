import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from pathlib import Path
import TrainingModel as tm

#Definiranje parametra
main_directory = "./16000_pcm_speeches"
speaker_audio = "./16000_pcm_speeches/speakers"
noise_files = "./16000_pcm_speeches/noise"

test_split = 0.1
shuffle_nb = 43
sample_r = 16000
scale = 0.5
size_of_batch = 128
number_of_epochs = 20


#Procesiranje zvuka(buke)
resample_audio_execute = (
    "for dir in `ls -1 " + noise_files + "`; do "
    "for file in `ls -1 " + noise_files + "/$dir/*.wav`; do "
    "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
    "$file | grep sample_rate | cut -f2 -d=`; "
    "if [ $sample_rate -ne 16000 ]; then "
    "ffmpeg -hide_banner -loglevel panic -y "
    "-i $file -ar 16000 temp.wav; "
    "mv temp.wav $file; "
    "fi; done; done"
)
os.system(resample_audio_execute)

def load_noise_files(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == sample_r:
        slices = int(sample.shape[0] / sample_r)
        sample = tf.split(sample[: slices * sample_r], slices)
        return sample
    else:
        print("Incorrect sampling rate")
        return None
    
path_to_noise_files_array = []
for subdir in os.listdir(noise_files):
    subdir_path = Path(noise_files) / subdir
    if os.path.isdir(subdir_path):
        path_to_noise_files_array += [
            os.path.join(subdir_path, filepath)
            for filepath in os.listdir(subdir_path)
            if filepath.endswith(".wav")
        ]
if not path_to_noise_files_array:
    raise RuntimeError("Could not find any files in this location")
print("Found requested files")

noise_array = []
for path in path_to_noise_files_array:
    sample = load_noise_files(path)
    if sample:
        noise_array.extend(sample)
noise_array = tf.stack(noise_array)

print("Noise files were successfully resampled")


#priprema baze
def paths_and_labels_to_files(audio_paths, labels):
    path_files = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_files = path_files.map(lambda x: path_to_audio(x))
    lable_files = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_files, lable_files))

def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, sample_r)
    return audio

def mix_noise_and_audio(audio, noises=None, scale=0.5):
    if noises is not None:
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        audio = audio + noise * prop * scale

    return audio

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

class_names = os.listdir(speaker_audio)
print(class_names,)

path_to_audi_files_array = []
array_of_lables = []
for label, name in enumerate(class_names):
    print("Speaker:",(name))
    dir_path = Path(speaker_audio) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    path_to_audi_files_array += speaker_sample_paths
    array_of_lables += [label] * len(speaker_sample_paths)

rng = np.random.RandomState(shuffle_nb)
rng.shuffle(path_to_audi_files_array)
rng = np.random.RandomState(shuffle_nb)
rng.shuffle(array_of_lables)


# Razdvajanje na treniranje i testiranje
number_of_valid_samples = int(test_split * len(path_to_audi_files_array))
train_audio_paths = path_to_audi_files_array[:-number_of_valid_samples]
train_labels = array_of_lables[:-number_of_valid_samples]


valid_audio_paths = path_to_audi_files_array[-number_of_valid_samples:]
valid_labels = array_of_lables[-number_of_valid_samples:]

train_variable = paths_and_labels_to_files(train_audio_paths, train_labels)
train_variable = train_variable.shuffle(buffer_size=size_of_batch * 8, seed=shuffle_nb).batch(
    size_of_batch
)

test_variable = paths_and_labels_to_files(valid_audio_paths, valid_labels)
test_variable = test_variable.shuffle(buffer_size=32 * 8, seed=shuffle_nb).batch(32)

train_variable = train_variable.map(
    lambda x, y: (mix_noise_and_audio(x, noise_array, scale=scale), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

train_variable = train_variable.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)

train_variable = train_variable.prefetch(tf.data.experimental.AUTOTUNE)

test_variable = test_variable.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
test_variable = test_variable.prefetch(tf.data.experimental.AUTOTUNE)
print("Training and testing set successfully made.")


#ccn model za treniranje 
if os.path.exists("model.h5"):
    histr = keras.models.load_model("model.h5")
    print("Model loaded successfully.")
else:
    print("Building new model.")
    histr = tm.training(sample_rate=sample_r, class_names=class_names, train_variable=train_variable, number_of_epochs=number_of_epochs, test_variable=test_variable)


#evaluacija performansi
print("Model evaluation.")
print(histr.evaluate(train_variable))

#primjena
speakers_number = 13

test_variable = paths_and_labels_to_files(valid_audio_paths, valid_labels)
test_variable = test_variable.shuffle(buffer_size=size_of_batch * 8, seed=shuffle_nb).batch(
    size_of_batch
)

test_variable = test_variable.map(
    lambda x, y: (mix_noise_and_audio(x, noise_array, scale=scale), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

for audios, array_of_lables in test_variable.take(1):
    ffts = audio_to_fft(audios)
    y_pred = histr.predict(ffts)
    rnd = np.random.randint(0, size_of_batch, speakers_number)
    audios = audios.numpy()[rnd, :, :]
    array_of_lables = array_of_lables.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(speakers_number):
        print(
            "Real speaker:\33{} {}\33[0m\tPredicted speaker:\33{} {}\33[0m".format(
                "[92m" if array_of_lables[index] == y_pred[index] else "[91m",
                class_names[array_of_lables[index]],
                "[92m" if array_of_lables[index] == y_pred[index] else "[91m",
                class_names[y_pred[index]],
            )
        )
