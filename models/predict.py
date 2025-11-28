import numpy as np

def predecir_paciente(modelo, scaler, valores):
    X = np.array(valores, dtype=float).reshape(1, -1)

    X_scaled = scaler.transform(X)

    prediccion = modelo.predict(X_scaled)[0]  

    estado = "Vive" if prediccion == 0 else "Muere"

    proba = modelo.predict_proba(X_scaled)[0]
    probabilidad_vive = float(proba[0] * 100)    
    probabilidad_muere = float(proba[1] * 100)  

    return {
        "estado": estado,
        "valor_crudo": int(prediccion),
        "probabilidad_vive": round(probabilidad_vive, 2),
        "probabilidad_muere": round(probabilidad_muere, 2)
    }
