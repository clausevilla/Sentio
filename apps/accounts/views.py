from datetime import datetime, timedelta

from django.contrib.auth import authenticate, login
from django.shortcuts import redirect, render


def login_view(request):
    #TODO NEED TO DOUBLE CHECK IMPLEMENTATION AFTER IMPLEMENTING USER LOGIN AND AUTHENTICATION
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('prediction:input')
        else:
            context = {'error': 'Invalid username of password'}
            return render(request, 'accounts/login.html', context)

    return render(request, 'accounts/login.html')


# @login_required
def logout_view(request):
    #TODO NEED TO MAKE SURE SESSION EXPIRES
    return render(request, 'home')


def register_view(request):
    #TODO NEED TO SEND USER REGISTRATION DATA
    return render(request, 'accounts/register.html', {})


def profile_view(request):
    #TODO NEED TO FETCH DATA AND SEND USER INFO
    return render(request, 'accounts/profile.html')


# @login_required
def history_view(request):
    # TODO IMPLEMENT THE METHOD
    # chart_data = get_chart_data(request.user, 'week')

    return render(request, 'accounts/history.html')



def get_chart_data(user, period='week'):

    # TODO GET REAL DATA FROM OUR DATABASE

    from .models import MentalHealthAnalysis  # ← YOUR MODEL NAME

    now = datetime.now()
    if period == 'week':
        start_date = now - timedelta(days=7)
        labels = [(start_date + timedelta(days=i)).strftime('%a') for i in range(7)]
    elif period == 'month':
        start_date = now - timedelta(days=30)
        labels = [
            (start_date + timedelta(days=i * 5)).strftime('%b %d') for i in range(6)
        ]
    else:
        start_date = now - timedelta(days=90)
        labels = [
            (start_date + timedelta(days=i * 15)).strftime('%b %d') for i in range(6)
        ]

    # TODO CHECK ALL FOLLOWING CODE, NEED TO USE REAL MENTAL STATE LABLES FROM DATABASE
    chart_data = {
        'labels': labels,
        'normal': [0] * len(labels),
        'depression': [0] * len(labels),
        'anxiety': [0] * len(labels),
        'stress': [0] * len(labels),
        'suicidal': [0] * len(labels),
        'bipolar': [0] * len(labels),
    }

    # TODO Get analyses from database 
    analyses = MentalHealthAnalysis.objects.filter(
        user=user, created_at__gte=start_date, created_at__lte=now
    ).order_by('created_at')

    # Count occurrences
    for analysis in analyses:
        if period == 'week':
            label = analysis.created_at.strftime('%a')
        elif period == 'month':
            days_diff = (analysis.created_at.date() - start_date.date()).days
            index = min(days_diff // 5, len(labels) - 1)
            label = labels[index]
        else:
            days_diff = (analysis.created_at.date() - start_date.date()).days
            index = min(days_diff // 15, len(labels) - 1)
            label = labels[index]

        if label in labels:
            idx = labels.index(label)
            state = analysis.mental_state  # ← YOUR FIELD NAME
            if state in chart_data:
                chart_data[state][idx] += 1

    return chart_data
