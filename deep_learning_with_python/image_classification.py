"""
Image classification using convolution networks on a small dataset.
"""

from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import os, shutil


# Data Collection (Split images into: train, validation and test)
train_range = (0, 1000)
validation_range = (1000, 1500)
test_range = (1500, 2000)

# source directory
source_dataset_dir = "/Users/Pavel/Desktop/Projects/TensorFlow/deep_learning_with_python/datasets/PetImages"

# train data directory
train_dir = os.path.join(source_dataset_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# validation data directory
validation_dir = os.path.join(source_dataset_dir, 'validation')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# test data directory
test_dir = os.path.join(source_dataset_dir, 'test')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


if False:
    # source directory
    #os.mkdir(source_dataset_dir)

    # train data directory
    os.mkdir(train_dir)
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)

    fnames = [f'{i}.jpg' for i in range(train_range[0], train_range[1])]
    for fname in fnames:
        src = os.path.join(source_dataset_dir, "Cat", fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'{i}.jpg'  for i in range(train_range[0], train_range[1])]
    for fname in fnames:
        src = os.path.join(source_dataset_dir, "Dog", fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # validation data directory
    os.mkdir(validation_dir)
    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)

    fnames = [f'{i}.jpg' for i in range(validation_range[0], validation_range[1])]
    for fname in fnames:
        src = os.path.join(source_dataset_dir, "Cat", fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'{i}.jpg'  for i in range(validation_range[0], validation_range[1])]
    for fname in fnames:
        src = os.path.join(source_dataset_dir, "Dog", fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # test data directory
    os.mkdir(test_dir)
    os.mkdir(test_cats_dir)
    os.mkdir(test_dogs_dir)

    fnames = [f'{i}.jpg' for i in range(test_range[0], test_range[1])]
    for fname in fnames:
        src = os.path.join(source_dataset_dir, "Cat", fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'{i}.jpg'  for i in range(test_range[0], test_range[1])]
    for fname in fnames:
        src = os.path.join(source_dataset_dir, "Dog", fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # print(len(os.listdir(train_cats_dir)))
    # print(len(os.listdir(train_dogs_dir)))


# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)  # rescale all images by 1/255
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(  # turns image files to batches of preprocessed tensors
    train_dir,
    target_size=(150, 150),  # rezize all images to 150*150
    batch_size=20,
    class_mode='binary'  # binary labels for binary crossentropy loss
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)



# Model architcture
model = Sequential()
model.add(
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(150, 150, 3))
)
model.add(
    MaxPool2D(pool_size=(2, 2))  # reduces output feauture map size twofold
)
model.add(
    Conv2D(64, kernel_size=(3, 3), activation="relu")
)
model.add(
    MaxPool2D(pool_size=(2, 2))
)
model.add(
    Conv2D(128, kernel_size=(3, 3), activation="relu")
)
model.add(
    MaxPool2D(pool_size=(2, 2))
)
model.add(
    Conv2D(128, kernel_size=(3, 3), activation="relu")
)
model.add(
    MaxPool2D(pool_size=(2, 2))
)
model.add(
    Flatten()
)
model.add(
    Dense(512, activation="relu")
)
model.add(
    Dense(1, activation="sigmoid")  # Binary classification between Cat or Dog
)


# Model compilation
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss=binary_crossentropy,  # sparse_categorical_crossentropy if integer tensor is used for labels
    metrics=binary_accuracy,  # i.e. hit ratio
)


# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # number of samples, i.e. steps_per_epoch (=100) * batch_size (=20) = 1000 * 2 (cats and dogs)
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)


# Save model
model.save('cats_and_dogs_small_1.h5')
