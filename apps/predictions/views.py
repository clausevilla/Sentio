import joblib
from django.shortcuts import redirect, render

# Loads the model
MODEL_PATH = 'ml_models/toy_models/LRmodel.pkl'
model = joblib.load(MODEL_PATH)

# Create your views here.


def input_view(request):
    if request.method == 'POST':
        user_text = request.POST.get('text')

        request.session['user_text'] = user_text

        return redirect('predictions:result')

    return render(request, 'predictions/input.html')


def result_view(request):
    user_text = request.session.get('user_text')

    if not user_text:
        return redirect('predictions:input')

    prediction, confidence = analyze_text(user_text)
    confidence = round(confidence * 100)
    print(confidence)

    return render(
        request,
        'predictions/result.html',
        {
            'analyzed_text': user_text,
            'mental_state_label': prediction,
            'confidence': confidence,
        },
    )


def analyze_text(analyzed_text):
    prediction = model.predict([analyzed_text])
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
    proba = model.predict_proba([analyzed_text])[0]
    confidence = max(proba)
    print(confidence)

    return (label, confidence)
