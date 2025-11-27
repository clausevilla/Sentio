from django.shortcuts import redirect, render

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

    prediction, confidence = analyze_text(user_text)
    confidence = round(confidence * 100)
    return render(
        request,
        'predictions/result.html',
        {
            'analyzed_text': user_text,
            'mental_state_label': prediction,
            'confidence': confidence,
        },
    )
