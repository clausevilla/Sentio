# Author: Marcus Berggren, Lian Shi, Karl Byland, Claudia Sevilla

from django.shortcuts import redirect, render

from apps.predictions.services import get_prediction_result

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

    # Get logged-in user or None
    user = request.user if request.user.is_authenticated else None
    # Saves the submission and prediction to the databast if the user is logged in
    prediction, confidence_percentage, anxiety_level, negativity_level, emotional_intensity, word_count, char_count = get_prediction_result(user, user_text)
    return render(
        request,
        'predictions/result.html',
        {
            'analyzed_text': user_text,
            'mental_state_label': prediction,
            'confidence': confidence_percentage,
            'anxiety_level': anxiety_level,
            'negativity_level': negativity_level,
            'emotional_intensity': emotional_intensity,
            'word_count': word_count,
            'char_count': char_count,
        },
    )
