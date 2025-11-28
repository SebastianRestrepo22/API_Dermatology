import numpy as np

def predecir_paciente(modelo, scaler, valores):
    # Convertimos valores a array y aseguramos la forma correcta
    X = np.array(valores, dtype=float).reshape(1, -1)

    # Normalización
    X_scaled = scaler.transform(X)

    # Predicción
    prediccion = modelo.predict(X_scaled)[0]   # 0 o 1 según tu modelo

    # Interpretación humana
    estado = "Vive" if prediccion == 0 else "Muere"

    # Probabilidades
    proba = modelo.predict_proba(X_scaled)[0]
    probabilidad_vive = float(proba[0] * 100)    # probabilidad clase 0
    probabilidad_muere = float(proba[1] * 100)   # probabilidad clase 1

    # Retorno para la API
    return {
        "estado": estado,
        "valor_crudo": int(prediccion),
        "probabilidad_vive": round(probabilidad_vive, 2),
        "probabilidad_muere": round(probabilidad_muere, 2)
    }
