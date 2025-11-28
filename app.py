from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="models/modelo_ann.tflite")
interpreter.allocate_tensors()

scaler = joblib.load("models/scaler.pkl")

app = Flask(__name__)

etiquetas = {
    0: "Leve",
    1: "Moderada",
    2: "Severa",
    3: "Crónico",
    4: "Recurrente",
    5: "Crítico"
}

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

    for campo in campos_requeridos:
        if campo not in data:
            return jsonify({"error": f"Falta el campo '{campo}' en el JSON"}), 400

    try:
        valores = [float(data[campo]) for campo in campos_requeridos]
    except ValueError:
        return jsonify({"error": "Todos los valores deben ser numéricos"}), 400

    valores_np = np.array(valores).reshape(1, -1)
    valores_scaled = scaler.transform(valores_np).astype(np.float32)

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, valores_scaled)
    interpreter.invoke()

    predicciones = interpreter.get_tensor(output_index)
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
        "mensaje": "Enviar POST a /api/predict con los valores"
    }), 200


if __name__ == "__main__":
    app.run()
