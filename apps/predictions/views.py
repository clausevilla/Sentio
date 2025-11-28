from django.shortcuts import redirect, render

from apps.predictions.models import PredictionResult, TextSubmission
from apps.predictions.services import analyze_text

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

    prediction, confidence, model_version = analyze_text(user_text)
    confidence = round(confidence * 100)
    # Get logged-in user or None
    user = request.user if request.user.is_authenticated else None

    submission = TextSubmission.objects.create(user=user, text_content=user_text)
    # Save to database
    PredictionResult.objects.create(
        submission=submission,
        model_version=model_version,
        stress_level=1,
        prediction=prediction,
        emotional_tone=confidence / 100,  # example: normalize back to 0-1 float
        social_confidence=0.1,  # placeholder, replace with actual values
        recommendations='Follow recommended steps',  # placeholder text
    )
    return render(
        request,
        'predictions/result.html',
        {
            'analyzed_text': user_text,
            'mental_state_label': prediction,
            'confidence': confidence,
        },
    )
