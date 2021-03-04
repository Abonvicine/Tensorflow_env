from preprossessing import basePreprocessing, baseNormalizer
from model import createModel
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import date

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os

hoje = date.today()
diaSemana = hoje.weekday()
hoje = str(hoje)

#carregar dados

baseDIR = "./files/datasets/DADOS SUDESTE"

historicoCarga = pd.read_csv(os.path.join(baseDIR,'SECO_2021-01-27_CARGAHIST.csv'),sep=";",decimal=",")
historicoTemperatura = pd.read_csv(os.path.join(baseDIR,'SECO_2021-01-27_TEMPHIST.csv'),sep=";",decimal=",")

#separar em treinamento e teste

#executar o pre-processamento

#carregar modelo

#rodar a previsão 

#calcular o MAPE

#printar os gráficos
