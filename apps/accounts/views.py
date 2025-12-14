# Author: Lian Shi
# Disclaimer: LLM has been used to help generate changepassword and delete account API endpoints.

import json
import logging
import re
from collections import defaultdict
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db import DatabaseError, IntegrityError
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_http_methods

from apps.predictions.models import PredictionResult, TextSubmission

from .forms import LoginForm, RegisterForm
from .models import UserConsent

# Set up logger
logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER: Error Page
# ==============================================================================


def render_error_page(request, message, retry_url, status=503):
    """
    Render a simple error page that auto-redirects after 10 seconds.
    Uses templates/accounts/error.html
    """
    return render(
        request,
        'accounts/error.html',
        {'message': message, 'retry_url': retry_url},
        status=status,
    )


# ==============================================================================
# REGISTRATION APIs
# ==============================================================================


@require_http_methods(['POST'])
@csrf_protect
def check_username_api(request):
    """
    API endpoint to check if a username is available.
    Used for real-time validation during registration.
    """
    try:
        data = json.loads(request.body)
        username = data.get('username', '').strip()

        errors = []

        if not username:
            return JsonResponse({'available': False, 'error': 'Username is required'})

        if len(username) < 3:
            errors.append('Username must be at least 3 characters')

        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            errors.append('Username can only contain letters, numbers, and underscores')

        if User.objects.filter(username__iexact=username).exists():
            errors.append('This username is already taken')

        if errors:
            return JsonResponse({'available': False, 'error': errors[0]})

        return JsonResponse({'available': True, 'message': 'Username is available'})

    except json.JSONDecodeError:
        return JsonResponse(
            {'available': False, 'error': 'Invalid request'}, status=400
        )
    except DatabaseError as e:
        logger.exception(f'Database error in check_username_api: {e}')
        return JsonResponse(
            {'available': False, 'error': 'Service temporarily unavailable'}, status=503
        )
    except Exception as e:
        logger.exception(f'Unexpected error in check_username_api: {e}')
        return JsonResponse(
            {'available': False, 'error': 'An unexpected error occurred'}, status=500
        )


@require_http_methods(['POST'])
@csrf_protect
def check_email_api(request):
    """
    API endpoint to check if an email is available.
    Used for real-time validation during registration.
    """
    try:
        data = json.loads(request.body)
        email = data.get('email', '').strip().lower()

        if not email:
            return JsonResponse({'available': False, 'error': 'Email is required'})

        email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_pattern, email):
            return JsonResponse(
                {'available': False, 'error': 'Please enter a valid email address'}
            )

        if User.objects.filter(email__iexact=email).exists():
            return JsonResponse(
                {
                    'available': False,
                    'error': 'An account with this email already exists',
                }
            )

        return JsonResponse({'available': True, 'message': 'Email is available'})

    except json.JSONDecodeError:
        return JsonResponse(
            {'available': False, 'error': 'Invalid request'}, status=400
        )
    except DatabaseError as e:
        logger.exception(f'Database error in check_email_api: {e}')
        return JsonResponse(
            {'available': False, 'error': 'Service temporarily unavailable'}, status=503
        )
    except Exception as e:
        logger.exception(f'Unexpected error in check_email_api: {e}')
        return JsonResponse(
            {'available': False, 'error': 'An unexpected error occurred'}, status=500
        )


@require_http_methods(['POST'])
@csrf_protect
def register_api(request):
    """
    API endpoint for AJAX registration.
    Returns JSON response instead of redirecting.
    """
    try:
        data = json.loads(request.body)

        form_data = {
            'username': data.get('username', ''),
            'email': data.get('email', ''),
            'password1': data.get('password1', ''),
            'password2': data.get('password2', ''),
            'consent': data.get('consent', False),
        }

        form = RegisterForm(form_data)

        if form.is_valid():
            user = form.save()

            UserConsent.objects.create(
                user=user, has_consented=True, consent_at=timezone.now()
            )

            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)
                return JsonResponse(
                    {
                        'success': True,
                        'message': f'Welcome {username}! Your account has been created successfully.',
                        'redirect': '/predictions/input',
                    }
                )

            return JsonResponse(
                {
                    'success': True,
                    'message': 'Account created successfully. Please log in.',
                    'redirect': '/accounts/login',
                }
            )
        else:
            errors = {}
            for field, error_list in form.errors.items():
                errors[field] = [str(e) for e in error_list]

            return JsonResponse({'success': False, 'errors': errors}, status=400)

    except json.JSONDecodeError:
        return JsonResponse(
            {'success': False, 'errors': {'__all__': ['Invalid request data']}},
            status=400,
        )
    except IntegrityError as e:
        logger.exception(f'Integrity error in register_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'errors': {'__all__': ['Username or email already exists']},
            },
            status=400,
        )
    except DatabaseError as e:
        logger.exception(f'Database error in register_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'errors': {'__all__': ['Service temporarily unavailable']},
            },
            status=503,
        )
    except Exception as e:
        logger.exception(f'Unexpected error in register_api: {e}')
        return JsonResponse(
            {'success': False, 'errors': {'__all__': ['An unexpected error occurred']}},
            status=500,
        )


# ==============================================================================
# PAGE VIEWS
# ==============================================================================


def register_view(request):
    """
    Handle user registration with consent
    """
    try:
        if request.user.is_authenticated:
            messages.info(request, 'You are already logged in.')
            return redirect('predictions:input')

        if request.method == 'POST':
            form = RegisterForm(request.POST)

            if form.is_valid():
                user = form.save()

                UserConsent.objects.create(
                    user=user, has_consented=True, consent_at=timezone.now()
                )

                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password1')
                user = authenticate(username=username, password=password)

                if user is not None:
                    login(request, user)
                    messages.success(
                        request,
                        f'Welcome {username}! Your account has been created successfully.',
                    )
                    return redirect('predictions:input')
            else:
                for field, errors in form.errors.items():
                    for error in errors:
                        messages.error(request, f'{error}')
        else:
            form = RegisterForm()

        context = {'form': form}
        return render(request, 'accounts/register.html', context)

    except DatabaseError as e:
        logger.exception(f'Database error in register_view: {e}')
        messages.error(request, 'Service temporarily unavailable. Please try again.')
        return render(request, 'accounts/register.html', {'form': RegisterForm()})
    except Exception as e:
        logger.exception(f'Unexpected error in register_view: {e}')
        messages.error(request, 'An unexpected error occurred. Please try again.')
        return render(request, 'accounts/register.html', {'form': RegisterForm()})


def login_view(request):
    """
    Handle user login with consent check
    """
    try:
        if request.user.is_authenticated:
            messages.info(request, 'You are already logged in.')
            return redirect('predictions:input')

        if request.method == 'POST':
            form = LoginForm(request, data=request.POST)

            if form.is_valid():
                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password')

                user = authenticate(request, username=username, password=password)

                if user is not None:
                    login(request, user)
                    if user.is_staff:
                        messages.success(request, f'Welcome back, {username}!')
                        return redirect('/')

                    try:
                        consent = UserConsent.objects.get(user=user)
                        if not consent.has_consented:
                            messages.warning(
                                request,
                                'Please review and accept our data processing terms to continue.',
                            )
                            return redirect('accounts:consent')
                    except UserConsent.DoesNotExist:
                        UserConsent.objects.create(user=user, has_consented=False)
                        messages.warning(
                            request,
                            'Please review and accept our data processing terms to continue.',
                        )
                        return redirect('accounts:consent')

                    messages.success(request, f'Welcome back, {username}!')

                    next_page = request.GET.get('next', 'predictions:input')
                    return redirect(next_page)
            else:
                messages.error(
                    request, 'Invalid username or password. Please try again.'
                )
        else:
            form = LoginForm()

        context = {'form': form}
        return render(request, 'accounts/login.html', context)

    except DatabaseError as e:
        logger.exception(f'Database error in login_view: {e}')
        messages.error(request, 'Service temporarily unavailable. Please try again.')
        return render(request, 'accounts/login.html', {'form': LoginForm()})
    except Exception as e:
        logger.exception(f'Unexpected error in login_view: {e}')
        messages.error(request, 'An unexpected error occurred. Please try again.')
        return render(request, 'accounts/login.html', {'form': LoginForm()})


@login_required(login_url='accounts:login')
def logout_view(request):
    """
    Handle user logout and end session
    """
    try:
        username = request.user.username
        logout(request)
        messages.success(
            request, f'You have been logged out successfully. See you soon, {username}!'
        )
        return redirect('accounts:login')
    except Exception as e:
        logger.exception(f'Error in logout_view: {e}')
        logout(request)
        return redirect('accounts:login')


@login_required(login_url='accounts:login')
def consent_view(request):
    """
    Handle data processing consent page
    """
    try:
        if request.method == 'POST':
            consent_given = request.POST.get('consent') == 'on'

            if consent_given:
                consent, created = UserConsent.objects.get_or_create(user=request.user)
                consent.give_consent()

                messages.success(
                    request, 'Thank you for consenting to our data processing terms.'
                )
                next_page = request.GET.get('next', 'predictions:input')
                return redirect(next_page)
            else:
                messages.error(
                    request, 'You must consent to continue using the service.'
                )

        return render(request, 'accounts/consent.html')

    except DatabaseError as e:
        logger.exception(f'Database error in consent_view: {e}')
        messages.error(request, 'Service temporarily unavailable. Please try again.')
        return render(request, 'accounts/consent.html')
    except Exception as e:
        logger.exception(f'Unexpected error in consent_view: {e}')
        messages.error(request, 'An unexpected error occurred. Please try again.')
        return render(request, 'accounts/consent.html')


def privacy_policy_view(request):
    """
    Display the privacy policy page
    """
    return render(request, 'accounts/privacy.html')


@login_required(login_url='accounts:login')
def profile_view(request):
    """
    Display user profile with account information and statistics
    """
    try:
        total_analyses = TextSubmission.objects.filter(user=request.user).count()

        consent_status = False
        consent_at = None
        try:
            consent = UserConsent.objects.get(user=request.user)
            consent_status = consent.has_consented
            consent_at = consent.consent_at
        except UserConsent.DoesNotExist:
            pass

        context = {
            'user': request.user,
            'total_analyses': total_analyses,
            'consent_status': consent_status,
            'consent_at': consent_at,
        }
        return render(request, 'accounts/profile.html', context)

    except DatabaseError as e:
        logger.exception(f'Database error in profile_view: {e}')
        return render_error_page(
            request,
            "We're having trouble loading your profile. Please try again.",
            '/accounts/profile/',
        )
    except Exception as e:
        logger.exception(f'Unexpected error in profile_view: {e}')
        return render_error_page(
            request,
            'Something went wrong. Please try again.',
            '/accounts/profile/',
            status=500,
        )


# ==============================================================================
# PASSWORD & ACCOUNT APIs
# ==============================================================================


@login_required(login_url='accounts:login')
@require_http_methods(['POST'])
def change_password_api(request):
    """
    API endpoint to change user password
    """
    try:
        data = json.loads(request.body)
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')

        if not current_password or not new_password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'missing_fields',
                    'message': 'Both current and new password are required',
                },
                status=400,
            )

        if not request.user.check_password(current_password):
            return JsonResponse(
                {
                    'success': False,
                    'error': 'incorrect_password',
                    'message': 'Current password is incorrect',
                },
                status=400,
            )

        if current_password == new_password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'same_password',
                    'message': 'New password must be different from current password',
                },
                status=400,
            )

        if len(new_password) < 8:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'weak_password',
                    'message': 'Password must be at least 8 characters long',
                },
                status=400,
            )

        request.user.set_password(new_password)
        request.user.save()

        update_session_auth_hash(request, request.user)

        return JsonResponse(
            {'success': True, 'message': 'Password changed successfully'}
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {'success': False, 'error': 'invalid_json', 'message': 'Invalid JSON data'},
            status=400,
        )
    except DatabaseError as e:
        logger.exception(f'Database error in change_password_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'Service temporarily unavailable',
            },
            status=503,
        )
    except Exception as e:
        logger.exception(f'Unexpected error in change_password_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'An unexpected error occurred',
            },
            status=500,
        )


@login_required(login_url='accounts:login')
@require_http_methods(['DELETE'])
def delete_all_data_api(request):
    """
    API endpoint to delete all user's analysis data (but keep account)
    Also revokes consent so user needs to re-consent
    """
    try:
        data = json.loads(request.body)
        password = data.get('password')

        if not password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'missing_password',
                    'message': 'Password is required',
                },
                status=400,
            )

        if not request.user.check_password(password):
            return JsonResponse(
                {
                    'success': False,
                    'error': 'incorrect_password',
                    'message': 'Incorrect password',
                },
                status=400,
            )

        deletion_result = TextSubmission.objects.filter(user=request.user).delete()
        deleted_count = deletion_result[0]

        try:
            consent = UserConsent.objects.get(user=request.user)
            consent.revoke_consent()
        except UserConsent.DoesNotExist:
            UserConsent.objects.create(
                user=request.user,
                has_consented=False,
                revoked_at=timezone.now(),
            )

        if request.user.is_staff:
            redirect_url = '/'
        else:
            redirect_url = '/accounts/consent/'

        return JsonResponse(
            {
                'success': True,
                'message': f'Successfully deleted {deleted_count} analyses. Your consent has been revoked.',
                'deleted_count': deleted_count,
                'consent_revoked': True,
                'redirect_url': redirect_url,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {'success': False, 'error': 'invalid_json', 'message': 'Invalid JSON data'},
            status=400,
        )
    except DatabaseError as e:
        logger.exception(f'Database error in delete_all_data_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'Service temporarily unavailable',
            },
            status=503,
        )
    except Exception as e:
        logger.exception(f'Unexpected error in delete_all_data_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'An unexpected error occurred',
            },
            status=500,
        )


@login_required(login_url='accounts:login')
@require_http_methods(['DELETE'])
def delete_account_api(request):
    """
    API endpoint to permanently delete user account and all associated data
    """
    try:
        data = json.loads(request.body)
        password = data.get('password')

        if not password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'missing_password',
                    'message': 'Password is required',
                },
                status=400,
            )

        if not request.user.check_password(password):
            return JsonResponse(
                {
                    'success': False,
                    'error': 'incorrect_password',
                    'message': 'Incorrect password',
                },
                status=400,
            )

        username = request.user.username

        request.user.delete()

        logout(request)

        return JsonResponse(
            {
                'success': True,
                'message': f'Account {username} has been permanently deleted',
            }
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {'success': False, 'error': 'invalid_json', 'message': 'Invalid JSON data'},
            status=400,
        )
    except DatabaseError as e:
        logger.exception(f'Database error in delete_account_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'Service temporarily unavailable',
            },
            status=503,
        )
    except Exception as e:
        logger.exception(f'Unexpected error in delete_account_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'An unexpected error occurred',
            },
            status=500,
        )


# ==============================================================================
# HISTORY VIEW
# ==============================================================================


@login_required(login_url='accounts:login')
def history_view(request):
    """
    Display user's analysis history with charts and statistics.
    """
    try:
        submissions = (
            TextSubmission.objects.filter(user=request.user)
            .select_related('predictionresult')
            .order_by('-submitted_at')
        )

        analyses = []
        for submission in submissions:
            try:
                prediction = submission.predictionresult
                raw_mental_state = prediction.mental_state or ''
                normalized_state = raw_mental_state.lower().strip()

                analyses.append(
                    {
                        'id': submission.id,
                        'text': submission.text_content,
                        'mental_state': normalized_state,
                        'get_mental_state_display': prediction.get_mental_state_display(),
                        'confidence': round(prediction.confidence * 100),
                        'anxiety_level': prediction.anxiety_level or 0,
                        'negativity_level': prediction.negativity_level or 0,
                        'emotional_intensity': prediction.emotional_intensity or 0,
                        'created_at': submission.submitted_at,
                        'recommendations': prediction.recommendations,
                    }
                )
            except PredictionResult.DoesNotExist:
                continue

        total_analyses = len(analyses)
        normal_count = sum(1 for a in analyses if a['mental_state'] == 'normal')
        concern_count = total_analyses - normal_count

        last_analysis = 'Never'
        if analyses:
            last_analysis = analyses[0]['created_at'].strftime('%b %d, %Y')

        state_counts = defaultdict(int)
        for analysis in analyses:
            state_counts[analysis['mental_state']] += 1

        state_distribution = {
            'normal': {
                'label': 'Normal',
                'count': state_counts.get('normal', 0),
                'percentage': round(
                    (state_counts.get('normal', 0) / total_analyses * 100)
                    if total_analyses > 0
                    else 0
                ),
            },
            'depression': {
                'label': 'Depression',
                'count': state_counts.get('depression', 0),
                'percentage': round(
                    (state_counts.get('depression', 0) / total_analyses * 100)
                    if total_analyses > 0
                    else 0
                ),
            },
            'stress': {
                'label': 'Stress',
                'count': state_counts.get('stress', 0),
                'percentage': round(
                    (state_counts.get('stress', 0) / total_analyses * 100)
                    if total_analyses > 0
                    else 0
                ),
            },
            'suicidal': {
                'label': 'Suicidal',
                'count': state_counts.get('suicidal', 0),
                'percentage': round(
                    (state_counts.get('suicidal', 0) / total_analyses * 100)
                    if total_analyses > 0
                    else 0
                ),
            },
        }

        paginator = Paginator(analyses, 10)
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)

        context = {
            'total_analyses': total_analyses,
            'normal_count': normal_count,
            'concern_count': concern_count,
            'last_analysis': last_analysis,
            'state_distribution': state_distribution,
            'analyses': page_obj,
            'page_obj': page_obj,
            'is_paginated': paginator.num_pages > 1,
        }

        return render(request, 'accounts/history.html', context)

    except DatabaseError as e:
        logger.exception(f'Database error in history_view: {e}')
        return render_error_page(
            request,
            "We're having trouble loading your history. Please try again.",
            '/accounts/history/',
        )
    except Exception as e:
        logger.exception(f'Unexpected error in history_view: {e}')
        return render_error_page(
            request,
            'Something went wrong. Please try again.',
            '/accounts/history/',
            status=500,
        )


# ==============================================================================
# CHART DATA API
# ==============================================================================


@login_required(login_url='accounts:login')
@require_http_methods(['GET'])
def chart_data_api(request):
    """
    API endpoint to fetch chart data for mental health trends over time.
    """
    try:
        period = request.GET.get('period', 'week')
        today = timezone.now().date()
        yesterday = today - timedelta(days=1)

        all_submissions = (
            TextSubmission.objects.filter(user=request.user)
            .select_related('predictionresult')
            .order_by('submitted_at')
        )

        if not all_submissions.exists():
            return JsonResponse(
                {
                    'labels': [],
                    'datasets': [
                        {
                            'label': 'Normal',
                            'data': [],
                            'borderColor': '#4A7C59',
                            'backgroundColor': 'rgba(74, 124, 89, 0.1)',
                        },
                        {
                            'label': 'Depression',
                            'data': [],
                            'borderColor': '#5B7C99',
                            'backgroundColor': 'rgba(91, 124, 153, 0.1)',
                        },
                        {
                            'label': 'Stress',
                            'data': [],
                            'borderColor': '#E07A5F',
                            'backgroundColor': 'rgba(224, 122, 95, 0.1)',
                        },
                        {
                            'label': 'Suicidal',
                            'data': [],
                            'borderColor': '#7B68A6',
                            'backgroundColor': 'rgba(123, 104, 166, 0.1)',
                        },
                    ],
                }
            )

        first_submission_date = all_submissions.first().submitted_at.date()

        today_has_data = all_submissions.filter(submitted_at__date=today).exists()

        if today_has_data:
            end_date = today
        else:
            end_date = yesterday

        if period == 'week':
            start_date = end_date - timedelta(days=6)
        elif period == 'month':
            start_date = end_date - timedelta(days=29)
        else:
            start_date = first_submission_date

        mental_states = ['normal', 'depression', 'stress', 'suicidal']
        daily_counts = defaultdict(lambda: {state: 0 for state in mental_states})

        for submission in all_submissions:
            sub_date = submission.submitted_at.date()

            if sub_date < start_date or sub_date > end_date:
                continue

            try:
                prediction = submission.predictionresult
                raw_state = prediction.mental_state or ''
                state = raw_state.lower().strip()

                if state in mental_states:
                    daily_counts[sub_date][state] += 1
            except PredictionResult.DoesNotExist:
                continue

        all_dates = []
        current = start_date
        while current <= end_date:
            all_dates.append(current)
            current += timedelta(days=1)

        total_days = len(all_dates)

        if total_days <= 14:
            labels = [d.strftime('%b %d') for d in all_dates]
            chart_data = {
                state: [daily_counts[d][state] for d in all_dates]
                for state in mental_states
            }
        elif total_days <= 60:
            labels, chart_data = bucket_data(all_dates, daily_counts, mental_states, 2)
        elif total_days <= 120:
            labels, chart_data = bucket_data(all_dates, daily_counts, mental_states, 4)
        else:
            labels, chart_data = bucket_data(all_dates, daily_counts, mental_states, 7)

        datasets = [
            {
                'label': 'Normal',
                'data': chart_data['normal'],
                'borderColor': '#4A7C59',
                'backgroundColor': 'rgba(74, 124, 89, 0.1)',
            },
            {
                'label': 'Depression',
                'data': chart_data['depression'],
                'borderColor': '#5B7C99',
                'backgroundColor': 'rgba(91, 124, 153, 0.1)',
            },
            {
                'label': 'Stress',
                'data': chart_data['stress'],
                'borderColor': '#E07A5F',
                'backgroundColor': 'rgba(224, 122, 95, 0.1)',
            },
            {
                'label': 'Suicidal',
                'data': chart_data['suicidal'],
                'borderColor': '#7B68A6',
                'backgroundColor': 'rgba(123, 104, 166, 0.1)',
            },
        ]

        return JsonResponse({'labels': labels, 'datasets': datasets})

    except DatabaseError as e:
        logger.exception(f'Database error in chart_data_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'Service temporarily unavailable',
            },
            status=503,
        )
    except Exception as e:
        logger.exception(f'Unexpected error in chart_data_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'An unexpected error occurred',
            },
            status=500,
        )


def bucket_data(all_dates, daily_counts, mental_states, group_days):
    """Helper function to bucket daily data into groups."""
    labels = []
    chart_data = {state: [] for state in mental_states}

    for i in range(0, len(all_dates), group_days):
        bucket_dates = all_dates[i : i + group_days]
        labels.append(bucket_dates[0].strftime('%b %d'))

        for state in mental_states:
            total = sum(daily_counts[d][state] for d in bucket_dates)
            chart_data[state].append(total)

    return labels, chart_data


# ==============================================================================
# DELETE ANALYSIS API
# ==============================================================================


@login_required(login_url='accounts:login')
@require_http_methods(['DELETE'])
def delete_analysis_api(request, analysis_id):
    """
    API endpoint to delete a single analysis.
    """
    try:
        prediction = PredictionResult.objects.select_related('submission').get(
            id=analysis_id, submission__user=request.user
        )

        submission = prediction.submission

        prediction.delete()
        submission.delete()

        return JsonResponse(
            {'success': True, 'message': 'Analysis deleted successfully'}
        )

    except PredictionResult.DoesNotExist:
        return JsonResponse(
            {'success': False, 'error': 'Analysis not found or access denied'},
            status=404,
        )
    except DatabaseError as e:
        logger.exception(f'Database error in delete_analysis_api: {e}')
        return JsonResponse(
            {'success': False, 'error': 'Service temporarily unavailable'},
            status=503,
        )
    except Exception as e:
        logger.exception(f'Unexpected error in delete_analysis_api: {e}')
        return JsonResponse(
            {'success': False, 'error': 'An unexpected error occurred'},
            status=500,
        )


# ==============================================================================
# EXPORT DATA API
# ==============================================================================


@login_required(login_url='accounts:login')
@require_http_methods(['GET'])
def export_data_api(request):
    """
    API endpoint to export all user data in JSON format
    For GDPR compliance and user data portability
    """
    try:
        user = request.user

        user_data = {
            'username': user.username,
            'email': user.email,
            'date_joined': user.date_joined.isoformat(),
        }

        try:
            consent = UserConsent.objects.get(user=user)
            consent_data = {
                'has_consented': consent.has_consented,
                'consent_at': consent.consent_at.isoformat()
                if consent.consent_at
                else None,
                'revoked_at': consent.revoked_at.isoformat()
                if consent.revoked_at
                else None,
            }
        except UserConsent.DoesNotExist:
            consent_data = None

        analysis_history = []
        submissions = (
            TextSubmission.objects.filter(user=user)
            .select_related('predictionresult')
            .order_by('-submitted_at')
        )

        for submission in submissions:
            try:
                prediction = submission.predictionresult
                analysis_history.append(
                    {
                        'id': submission.id,
                        'text': submission.text_content,
                        'mental_state': prediction.mental_state,
                        'confidence': prediction.confidence,
                        'anxiety_level': prediction.anxiety_level,
                        'negativity_level': prediction.negativity_level,
                        'emotional_intensity': prediction.emotional_intensity,
                        'recommendations': prediction.recommendations,
                        'submitted_at': submission.submitted_at.isoformat(),
                        'predicted_at': prediction.predicted_at.isoformat(),
                    }
                )
            except PredictionResult.DoesNotExist:
                continue

        export_data = {
            'user_profile': user_data,
            'consent_data': consent_data,
            'analysis_history': analysis_history,
            'exported_at': timezone.now().isoformat(),
        }

        json_data = json.dumps(export_data, indent=4, ensure_ascii=False)
        response = HttpResponse(json_data, content_type='application/json')
        response['Content-Disposition'] = (
            f'attachment; filename="{user.username}_data_export.json"'
        )
        return response

    except DatabaseError as e:
        logger.exception(f'Database error in export_data_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'Service temporarily unavailable',
            },
            status=503,
        )
    except Exception as e:
        logger.exception(f'Unexpected error in export_data_api: {e}')
        return JsonResponse(
            {
                'success': False,
                'error': 'server_error',
                'message': 'An unexpected error occurred',
            },
            status=500,
        )
