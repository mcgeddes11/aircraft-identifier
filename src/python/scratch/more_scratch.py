from keras.models import Sequential
from keras.layers import Dense, ReLU, Activation, Dropout
from keras.layers.normalization import BatchNormalization

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(94, input_dim=94, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(16, kernel_initializer="normal", activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Dense(1, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model





