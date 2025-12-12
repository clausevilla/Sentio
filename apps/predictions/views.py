# Author: Marcus Berggren, Lian Shi, Karl Byland, Claudia Sevilla

import logging
import json
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render

from apps.predictions.services import get_prediction_result

logger = logging.getLogger(__name__)


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

    user = request.user if request.user.is_authenticated else None

    (
        prediction,
        confidence_percentage,
        recommendations,
        anxiety_level,
        negativity_level,
        emotional_intensity,
        word_count,
        char_count,
        all_confidences,  # ← ADD THIS
    ) = get_prediction_result(user, user_text)

    return render(
        request,
        'predictions/result.html',
        {
            'analyzed_text': user_text,
            'mental_state_label': prediction,
            'confidence': confidence_percentage,
            'recommendations': recommendations,
            'anxiety_level': anxiety_level,
            'negativity_level': negativity_level,
            'emotional_intensity': emotional_intensity,
            'word_count': word_count,
            'char_count': char_count,
            'all_confidences': all_confidences,
        },
    )


def strings(request):
    path = Path(settings.BASE_DIR) / 'apps' / 'predictions' / 'data' / 'strings.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return JsonResponse(data)
