import pandas as pd

def basePreprocessing(carga,temperatura):
    dataset = carga.merge(temperatura)
    dataset = dataset.drop(["Min"], axis = 1 )
    dataset = dataset.rename({'Ano': 'Year', 'Mes': 'Month',"Dia":"Day"}, axis=1)
    dataset["Date"] = pd.to_datetime(dataset[["Year","Month","Day"]])
    dataset["Weekday"] = dataset["Date"].dt.dayofweek
    
    return dataset

def baseNormalizer(dataset, normalizer, normalize_fit = True):
    
    normalizadorCarga = normalizer
    normalizadorTemp = normalizer

    if normalize_fit:
        dataset[["Carga"]] = normalizadorCarga.fit_transform(dataset[["Carga"]])
        dataset[["Temperatura"]] = normalizadorTemp.fit_transform(dataset[["Temperatura"]])

    dummies = pd.get_dummies(dataset["Weekday"],prefix="Dia")
    dataset = dataset.join(dummies)
    dataset = dataset.drop(["Date","Weekday"], axis = 1 )
    dataset = dataset.dropna()

    return dataset , normalizadorCarga, normalizadorTemp