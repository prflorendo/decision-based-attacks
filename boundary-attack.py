from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
import os
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from PIL import Image
from keras.preprocessing import image

# train classification model
def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (X_train, y_train), (X_test, y_test)

# TODO: save this to file so it doesn't retrain lol
def create_mnist_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_mnist_sample(sample):
    return sample

# no idea where this comes from tbh
RESNET_MEAN = np.array([103.939, 116.779, 123.68])

def orthogonal_perturbation(delta, prev_sample, target_sample):
# orthogonal step
    perturb = np.random.randn(1, 224, 224, 3)
    perturb /= np.linalg.norm(perturb, axis=(1, 2))
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))

# Project perturbation onto sphere around target
    diff = (target_sample - prev_sample).astype(np.float32) # Orthorgonal vector to sphere surface
    diff /= get_diff(target_sample, prev_sample) # Orthogonal unit vector

# We project onto the orthogonal then subtract from perturb
# to get projection onto sphere surface
    perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff)**2) * diff

# Check overflow and underflow
    overflow = (prev_sample + perturb) - 255 + RESNET_MEAN
    isOverflow = overflow > 0
    perturb -= overflow * isOverflow
    underflow = -RESNET_MEAN
    isUnderflow = underflow > 0
    perturb += underflow * isUnderflow
    return perturb


def forward_perturbation(epsilon, prev_sample, target_sample):
    # forward step
    perturb = target_sample - prev_sample
    perturb = perturb.astype(np.float32)
    perturb *= epsilon
    return perturb


def get_converted_prediction(sample, classifier):
	"""
	The original sample is dtype float32, but is converted
	to uint8 when exported as an image. The loss of precision
	often causes the label of the image to change, particularly
	because we are very close to the boundary of the two classes.
	This function checks for the label of the exported sample
	by simulating the export process.
	"""
	sample = (sample + RESNET_MEAN).astype(np.uint8).astype(np.float32) - RESNET_MEAN
	label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
	return label


def save_image(sample, classifier, folder):
	"""Export image file."""
	label = get_converted_prediction(np.copy(sample), classifier)
	sample = sample[0]
	# Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
	sample += RESNET_MEAN
	sample = sample[..., ::-1].astype(np.uint8)
	# Convert array to image and save
	sample = Image.fromarray(sample)
	id_no = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	# Save with predicted label for image (may not be adversarial due to uint8 conversion)
	sample.save(os.path.join("images", folder, "{}_{}.png".format(id_no, label)))


def preprocess(sample_path):
    # """Load and preprocess image file."""
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x


def get_diff(sample_1, sample_2):
	"""Channel-wise norm of difference between samples."""
	return np.linalg.norm(sample_1 - sample_2, axis=(1, 2))

# actual attack
def boundary_attack():
    # Load model, images and other parameters
    (X_train, y_train), (_, _) = load_mnist_data()
    classifier = create_mnist_model()
    classifier.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    # initial_sample = preprocess_mnist_sample(X_train[0])
    # target_sample = preprocess_mnist_sample(X_train[1])

    # set initial and target samples
    initial_sample = preprocess('images/original/0.png')
    target_sample = preprocess('images/original/1.png')

    # Create folder for images
    folder = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.mkdir(os.path.join("images", folder))
    save_image(np.copy(initial_sample), classifier, folder)
    initial_class = np.argmax(classifier.predict(np.expand_dims(initial_sample, axis=0)))
    target_class = np.argmax(classifier.predict(np.expand_dims(target_sample, axis=0)))

    # Initialize parameters
    # Not super specific, dynamically adjusted later
    adversarial_sample = initial_sample
    n_steps = 500
    n_calls = 500
    epsilon = 5
    delta = 5

    # Find boundary
    while True:
        step = forward_perturbation(epsilon, adversarial_sample, target_sample)
        boundary = adversarial_sample + step
        prediction = classifier.predict(boundary)
        n_calls += 1

        if np.argmax(prediction) == initial_class:
            adversarial_sample = boundary
            break

        else:
            # baby steps towards boundary
            epsilon *= 0.9

    # attack iterations
    while True:
        print("Step #{}...".format(n_steps))

        # Orthogonal step
        print("\tDelta step...")
        d_step = 0
        while True:
            d_step += 1
            print("\t#{}".format(d_step))
            trial_samples = []

            # walk around, see how many are correctly classified
            for i in np.arange(10):
                step = orthogonal_perturbation(delta, adversarial_sample, target_sample)
                trial_sample = adversarial_sample + step
                trial_samples.append(trial_sample)

            predictions = classifier.predict(trial_sample)
            n_calls += 10
            predictions = np.argmax(predictions, axis=1)
            d_score = np.mean(predictions == initial_class)

            # dynamic adjustment of delta
            # we expect d to be about 0.5, change it if weird
            if d_score > 0.0:
                if d_score < 0.25:
                    delta *= 0.9
                elif d_score > 0.75:
                    delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == initial_class)[0][0]]
                break
            else:
                delta *= 0.9

        # Forward step
        print("\tEpsilon step...")
        e_step = 0
        while True:
            e_step += 1
            print("\t#{}".format(e_step))
            step = forward_perturbation(epsilon, adversarial_sample, target_sample)
            trial_sample = adversarial_sample + step
            prediction = classifier.predict(trial_sample)
            n_calls += 1

            # bigger step if we're still in the same class
            if np.argmax(prediction) == initial_class:
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            # break if it takes too long
            elif e_step > 500:
                    break
            # else needs smaller step
            else:
                epsilon *= 0.5

        n_steps += 1
        chkpts = [1, 5, 10, 50, 100, 500]
        if (n_steps in chkpts) or (n_steps % 500 == 0):
            print("{} steps".format(n_steps))
            save_image(np.copy(adversarial_sample), classifier, folder)
        diff = np.mean(get_diff(adversarial_sample, target_sample))

        # pretty much converged or too many steps
        if diff <= 1e-3 or e_step > 500:
            print("{} steps".format(n_steps))
            print("Mean Squared Error: {}".format(diff))
            save_image(np.copy(adversarial_sample), classifier, folder)
            break

        print("Mean Squared Error: {}".format(diff))
        print("Calls: {}".format(n_calls))
        print("Attack Class: {}".format(initial_class))
        print("Target Class: {}".format(target_class))
        print("Adversarial Class: {}".format(np.argmax(prediction)))


if __name__ == "__main__":
	boundary_attack()
