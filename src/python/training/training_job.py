# This training assumes that the images have been downloaded and extracted from this URL:
# http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz

import pandas, numpy, os
from utils import save_data, load_data, create_folder
from sklearn.model_selection import train_test_split
from shutil import copy2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


def process_for_input(input_folder, output_folder, class_mapping):
    # This function takes the raw training set and rewrites it into the format that the Keras ImageDataGenerator expects
    # (ie.  test/train at first level, with a subfolder for each class)
    l = len(class_mapping)
    checkfile = os.path.join(input_folder,"__complete_check__.txt")
    if not os.path.exists(checkfile):
        for ix, row in class_mapping.iterrows():
            filename_to_copy = os.path.join(input_folder,row["image_id"] + ".jpg")
            destination_path = os.path.join(output_folder, row["data_segment"], row["label"], row["image_id"] + ".jpg")
            create_folder(destination_path)
            copy2(filename_to_copy,destination_path)
            print "{} / {} complete".format(ix, l)
        save_data("",checkfile)
    return True

def train_from_scratch(input_folder, output_folder, classmap):
    # Train a simple model from scratch

    # dimensions of our images.
    img_width, img_height = 150, 150

    train_data_dir = os.path.join(input_folder,"train")
    validation_data_dir = os.path.join(input_folder,"test")
    nb_train_samples = len(classmap[classmap["data_segment"] == "train"])
    nb_validation_samples = len(classmap[classmap["data_segment"] == "test"])
    nb_classes = len(numpy.unique(classmap["label"].values))
    epochs = 50
    batch_size = 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')

    return True

def bootstrap_mobilenet():
    # Bootstrap the MobileNet network and use transfer learning to train a new model on our aircraft data
    pass

def bootstrap_inceptionv3():
    # Bootstrap the InceptionV3 network and use transfer learning to train a new model on our aircraft data
    pass

def parse_classmap(input_list):
    classmap = [x.split(" ") for x in input_list]
    image_id = [x[0] for x in classmap]
    label = ["_".join(x[1:len(x)]) for x in classmap]
    df = pandas.DataFrame(data=image_id, columns=["image_id"])
    df["label"] = label
    return df



if __name__ == "__main__":

    # Define folders
    base_folder = "/home/user/data/aircraft_indentifier/fgvc-aircraft-2013b/data"
    image_folder = os.path.join(base_folder, "images")
    data_output_folder = "/home/user/data/aircraft_indentifier/model_input_data"

    # Read in image metadata and parse
    train_data_map = parse_classmap(load_data(os.path.join(base_folder,"images_family_train.txt")).split("\n"))
    test_data_map = parse_classmap(load_data(os.path.join(base_folder, "images_family_test.txt")).split("\n"))
    trainval_data_map = parse_classmap(load_data(os.path.join(base_folder, "images_family_trainval.txt")).split("\n"))
    all_datamap = pandas.concat((train_data_map,test_data_map,trainval_data_map),axis=0)
    # Remove any empty rows
    all_datamap = all_datamap[all_datamap["image_id"] != ""]

    # Split the data in a stratified way
    X_train, X_test, y_train, y_test = train_test_split(all_datamap, all_datamap["label"], test_size=0.2)
    X_train["data_segment"] = "train"
    X_test["data_segment"] = "test"
    all_datamap = pandas.concat((X_train,X_test), axis=0).reset_index(drop=True)
    all_datamap["label"] = [x.replace("/","_").replace("-","_") for x in all_datamap["label"].values]

    # copy to the input data to the structure Keras expects
    process_for_input(input_folder=image_folder,output_folder=data_output_folder, class_mapping=all_datamap)

    # Try building a network from scratch
    output_folder = os.path.join("/home/user/data/aircraft_indentifier/models/new_model")
    create_folder(os.path.join(output_folder,"foo.txt"))
    train_from_scratch(input_folder=data_output_folder, output_folder=output_folder, classmap=all_datamap)
