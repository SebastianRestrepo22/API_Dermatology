from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar modelo y scaler
modelo = load_model("models/modelo_ann.h5")
scaler = joblib.load("models/scaler.pkl")

app = Flask(__name__)

# Etiquetas de salida (según tu dataset de psoriasis o clasificación de enfermedad)
etiquetas = {
    0: "Leve",
    1: "Moderada",
    2: "Severa",
    3: "Crónico",
    4: "Recurrente",
    5: "Crítico"
}

# Lista de características esperadas en la misma estructura usada en entrenamiento
campos_requeridos = [
    "eritema", "elevacion_borde", "escala_difusa", "puntuacion_folicular", "eritrodermia",
    "pustulas", "placas", "picazon", "dolor", "lesiones_orales", "poliadenopatia",
    "pelo", "perdida_peso", "inflamacion_unas", "piel_lineas", "escamas_ricas",
    "crustas", "exudado", "pustulas_ricas", "infiltracion", "dermografismo", "kobrner",
    "area_afectada", "blanqueamiento", "familia", "histologia_espongiosis",
    "histologia_parakeratosis", "histologia_acantosis", "histologia_elongacion",
    "histologia_hyperqueratosis", "histologia_puntaje_epi", "histologia_spongiosis",
    "histologia_infiltrado", "edad"
]

@app.route("/api/predict", methods=["POST"])
def calcular_prediccion_endpoint():
    if not request.is_json:
        return jsonify({"error": "El contenido debe ser JSON"}), 400

    data = request.get_json()
    print("JSON recibido:", data)

    # Validación de campos
    for campo in campos_requeridos:
        if campo not in data:
            return jsonify({"error": f"Falta el campo '{campo}' en el JSON"}), 400
    
    try:
        valores = [float(data[campo]) for campo in campos_requeridos]
    except ValueError:
        return jsonify({"error": "Todos los valores deben ser numéricos"}), 400

    # Transformación y predicción
    valores_np = np.array(valores).reshape(1, -1)
    valores_scaled = scaler.transform(valores_np)
    
    predicciones = modelo.predict(valores_scaled)
    clase_predicha = int(np.argmax(predicciones))
    resultado = etiquetas[clase_predicha]

    return jsonify({
        "resultado": resultado,
        "clase_numero": clase_predicha,
        "probabilidades": predicciones.tolist()
    }), 200


@app.route("/api/predict/ejemplo", methods=["GET"])
def ejemplo():
    return jsonify({
        "features_esperadas": campos_requeridos,
        "mensaje": "Enviar POST a /api/modelo con los valores numéricos en formato JSON"
    }), 200


if __name__ == "__main__":
    app.run()
