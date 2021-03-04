from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import date

def createModel(input_shape,input_number):
    model = Sequential()

    model.add(LSTM(units = 100, return_sequences = True,input_shape = (input_shape,input_number)))
    model.add(Dropout(0.5))

    # model.add(LSTM(units = 50, return_sequences = True))
    # model.add(Dropout(0.3))

    model.add(LSTM(units = 30))
    model.add(Dropout(0.5))

    model.add(Dense(units = 1, activation= "linear"))

    model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = ["mean_absolute_error"])
    
    return model

def setCallbacks():

    callbacks = {
        "es": EarlyStopping(monitor = "loss", min_delta = 1e-10, patience = 10, verbose = 1),
        "rlr": ReduceLROnPlateau(monitor = "loss", factor = 0.2, patience = 5, verbose = 1),
        "mcp": ModelCheckpoint(filepath=f"./files/models/pesos-{date.today()}.h5", monitor = "loss", save_best_only = True, verbose=1)
    }
    return callbacks

def loadModel(date:date):

    day = str(date.day).zfill(2)
    month = str(date.month).zfill(2)
    year = str(date.year)

    model = keras.models.load_model(f'./files/models/pesos-{year}-{month}-{day}.h5')
    return model