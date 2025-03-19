import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Archivo de Excel donde se guardarán los datos
archivo_excel = "valores_usuario.xlsx"

# Intentar cargar el archivo, si no existe se crea un DataFrame vacío
try:
    df = pd.read_excel(archivo_excel)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Nombre", "Personalidad", "Visión", "Política", "Cultura", "Creencias", "Compatibilidad"])

# Opciones de valores
opciones = {
    "Personalidad": ["Extrovertido", "Introvertido", "Atrevido", "Tímido", "Activo", "Sedentario", "Fiestero", "Casero"],
    "Visión": ["Ambicioso", "Desinteresado", "Lujoso", "Austero", "Excéntrico", "Retacado", "Materialista", "Idealista"],
    "Política": ["Izquierda", "Derecha", "Radical", "Apolítico", "Patriota", "Anarquista", "Libertario", "Progresista"],
    "Cultura": ["Liberal", "Conservador", "Globalista", "Costumbrista", "Ambientalista", "Antiambientalista", "Tradicionalista", "Feminista o LGTBIQ+"],
    "Creencias": ["Espiritual", "Agnóstico", "Musulmán", "Budista", "Creyente (católico o cristiano)", "Ateo", "Hinduista", "Raíces costumbristas"]
}

# Ingreso de datos del usuario
nombre_usuario = input("Ingresa tu nombre: ")
respuestas_usuario = {}
valores_numericos = {}

for categoria, opciones_lista in opciones.items():
    print(f"\nSelecciona una opción para {categoria}:")
    for i, opcion in enumerate(opciones_lista):
        print(f"{i+1}. {opcion}")

    seleccion = int(input(f"Elige un número (1-{len(opciones_lista)}): ")) - 1
    respuesta = opciones_lista[seleccion]
    respuestas_usuario[categoria] = respuesta

    # Asignar valor numérico entre 0 y 10
    valores_numericos[categoria] = (seleccion / (len(opciones_lista) - 1)) * 10
  

# Función para calcular compatibilidad (30%)
def calcular_compatibilidad(valores):
    promedio = np.mean(list(valores.values()))
    compatibilidad = (promedio / 10) * 100
    return round(compatibilidad * 0.3, 2)

# Calcular compatibilidad
compatibilidad_30 = calcular_compatibilidad(valores_numericos)

# Guardar datos en el DataFrame
df = pd.concat([df, pd.DataFrame([{"Nombre": nombre_usuario, **respuestas_usuario, "Compatibilidad": compatibilidad_30}])], ignore_index=True)

# Guardar en Excel
df.to_excel(archivo_excel, index=False)

# Función para graficar el hexágono
"""def generar_hexagono(valores, etiquetas, compatibilidad):
    angles = np.linspace(0, 2 * np.pi, len(valores), endpoint=False).tolist()
    valores += valores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, valores, color='b', alpha=0.3)
    ax.plot(angles, valores, color='b', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(etiquetas)

    plt.title("Perfil de Valores del Usuario", pad=20)
    plt.figtext(0.5, 0.02, f"Compatibilidad: {compatibilidad:.2f}% sobre 30%", ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    plt.show()"""


# Graficar el espectro hexagonal
labels = list(valores_numericos.keys())
values = list(valores_numericos.values())
print(f"\nValores numéricos: {values}")
print(f"Compatibilidad (30%): {compatibilidad_30:.2f}%")
#generar_hexagono(values, labels, compatibilidad_30)

# Modelo de aprendizaje automático con validación cruzada
X = df.apply(lambda row: [opciones[categoria].index(row[categoria]) if row[categoria] in opciones[categoria] else 0 for categoria in ["Personalidad", "Visión", "Política", "Cultura", "Creencias"]], axis=1, result_type='expand')

y = df["Compatibilidad"].fillna(0)

kf = KFold(n_splits=5, shuffle=True, random_state=None)

errores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    errores.append(mean_squared_error(y_test, predicciones))

# Promedio del error cuadrático medio
error_promedio = np.mean(errores)

print(f"\nError cuadrático medio promedio (MSE): {error_promedio:.4f}")
