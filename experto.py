import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Definimos las categorías y sus correspondientes números
categorias = {
    'uso': {'personal': 1, 'familiar': 2, 'comercial': 3},
    'tamanio_familia': {'1 persona': 1, '2 personas': 2, '3 personas o más': 3},
    'presupuesto': {'bajo': 1, 'medio': 2, 'alto': 3},
    'distancia_dia': {'Menos de 10 km': 1, 'Entre 10 y 50 km': 2, 'Más de 50 km': 3},
    'tipo_ciudad': {'urbano': 1, 'suburbano': 2, 'rural': 3},
    'prioridades': {'precio': 1, 'rendimiento': 2, 'seguridad': 3},
    'tipo_carroceria': {'sedán': 1, 'SUV': 2, 'camioneta': 3},
}

# Nombres de modelos de automóviles
nombres_modelos = {
    1: 'Toyota Corolla',
    2: 'Honda Civic',
    3: 'Ford Escape',
    8: 'Honda Accord',
    9: 'Nissan Altima',
    10: 'Chevrolet Malibu',
    11: 'Volkswagen Passat',
    12: 'Honda CR-V',
    13: 'Toyota RAV4',
    14: 'Mazda CX-5',
    15: 'Ford Explorer',
    16: 'Hyundai Tucson',
    17: 'Kia Sportage',
    18: 'Subaru Forester',
    19: 'Toyota Tacoma',
    20: 'Chevrolet Colorado',
    21: 'Ram 1500 Classic',
    22: 'Ford Ranger',
    23: 'Nissan Frontier',
    24: 'Honda HR-V',
    25: 'Toyota Corolla Cross',
    26: 'Mazda CX-30',
    27: 'Hyundai Kona',
    28: 'Kia Seltos',
    29: 'Subaru Crosstrek',
    30: 'Toyota Sienna',
    31: 'Honda Odyssey',
    32: 'Chrysler Pacifica',
    33: 'Dodge Grand Caravan',
    34: 'Kia Carnival',
    35: 'Hyundai Palisade',
    36: 'Chevrolet Tahoe',
    37: 'GMC Yukon',
    38: 'Cadillac Escalade',
    39: 'Ford Expedition',
    40: 'Lincoln Navigator',
    41: 'Toyota 4Runner',
    42: 'Nissan Pathfinder',
    43: 'Honda Pilot',
    44: 'Mazda CX-9',
    45: 'Hyundai Santa Fe',
    46: 'Kia Telluride',
    47: 'Subaru Ascent',
    48: 'Toyota Highlander',
    49: 'Chevrolet Tahoe Hybrid',
    50: 'GMC Yukon Hybrid',
}

# Preguntas
preguntas = [
    '**1.** ¿Cuál es el uso principal del auto? (1 - personal, 2 - familiar, 3 - comercial)\n',
    '**2.** ¿Cuál es el tamaño de la familia? (1 - 1 persona, 2 - 2 personas, 3 - 3 personas o más)\n',
    '**3.** ¿Cuál es el presupuesto? (1 - bajo, 2 - medio, 3 - alto)\n',
    '**4.** ¿Cuál es la distancia promedio que recorres al día? (1 - Menos de 10 km, 2 - Entre 10 y 50 km, 3 - Más de 50 km)\n',
    '**5.** ¿En qué tipo de ciudad vives? (1 - urbano, 2 - suburbano, 3 - rural)\n',
    '**6.** ¿Cuáles son tus prioridades al comprar un auto? (1 - Precio, 2 - Rendimiento, 3 - Seguridad)\n',
    '**7.** ¿Cuáles son tus preferencias de tipo de carrocería? (1 - Sedán, 2 - SUV, 3 - Camioneta)\n',
]

# Obtenemos las respuestas a las preguntas del usuario
respuestas = []
for pregunta in preguntas:
    respuesta = int(input(pregunta))
    respuestas.append(respuesta)

# Convertimos las respuestas a un DataFrame con nombres de características
respuestas_df = pd.DataFrame([respuestas], columns=categorias.keys())

# Cargamos los datos de entrenamiento (reemplaza 'data_recomendaciones.csv' con tu propio archivo si es diferente)
data_recomendaciones_with_names = pd.read_csv('data_recomendaciones.csv')

# Creamos un modelo de Bosque Aleatorio (Random Forest) preentrenado
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenamos el modelo
model.fit(data_recomendaciones_with_names.drop(columns=['modelo']), data_recomendaciones_with_names['modelo'])

# Obtenemos las probabilidades de cada modelo
probabilidades = model.predict_proba(respuestas_df)

# Seleccionamos los modelos con probabilidades más altas
modelos_seleccionados = [nombre for id_modelo, nombre in nombres_modelos.items() if id_modelo in np.argmax(probabilidades, axis=1)]

# Buscamos todos los modelos que coinciden con las características
modelos_coincidentes = [(id_modelo, nombre) for id_modelo, nombre in nombres_modelos.items() if model.predict([respuestas])[0] == id_modelo]

# Mostramos la recomendación o sugerimos almacenar la respuesta
if len(modelos_coincidentes) == 0:
    print('Lo siento, no hay coincidencias con tus respuestas.')
    print('Aquí tienes la lista de modelos disponibles:')
    for id_modelo, nombre_modelo in nombres_modelos.items():
        print(f"{id_modelo}: {nombre_modelo}")

    # Preguntamos si quieren almacenar la respuesta
    almacenar_respuesta = input('¿Deseas almacenar esta respuesta para entrenar el modelo? (Sí/No): ')
    if almacenar_respuesta.lower() == 'si':
        # Almacenamos la respuesta en el conjunto de entrenamiento
        nueva_fila = respuestas + [input('Por favor, ingresa el modelo recomendado: ')]
        data_recomendaciones_with_names.loc[len(data_recomendaciones_with_names)] = nueva_fila
        data_recomendaciones_with_names.to_csv('data_recomendaciones.csv', index=False)
        print('¡Respuesta almacenada! Puedes usarla para entrenar el modelo la próxima vez.')

else:
    print('Según tus respuestas, el experto recomienda los siguientes modelos:')
    for id_modelo, nombre_modelo in modelos_coincidentes:
        print(nombre_modelo)
        # Mostramos las características que más influyeron en la predicción
        importances = model.feature_importances_
        caracteristicas = list(categorias.keys())
        influencia_caracteristicas = sorted(list(zip(caracteristicas, importances)), key=lambda x: x[1], reverse=True)
        print('Motivos de la recomendación:')
        for caracteristica, influencia in influencia_caracteristicas:
            print(f"- {caracteristica}: {influencia}")
