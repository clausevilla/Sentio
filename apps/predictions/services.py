import joblib

# Loads the model
MODEL_PATH = 'ml_pipeline/toy_models/LRmodel.pkl'
MODEL = joblib.load(MODEL_PATH)


def analyze_text(analyzed_text):
    prediction = MODEL.predict([analyzed_text])
    outputMap = {
        0: 'Normal',
        1: 'Depression',
        2: 'Suicidal',
        3: 'Anxiety',
        4: 'Stress',
        5: 'Bipolar',
        6: 'Personality disorder',
    }
    label = outputMap.get(int(prediction[0]))
    proba = MODEL.predict_proba([analyzed_text])[0]
    confidence = max(proba)

    return (label, confidence)
