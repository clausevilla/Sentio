# Author: Lian Shi
# Disclaimer: LLM has been used to help generate changepassword and delete account API endpoints.

import json
from collections import defaultdict
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from apps.predictions.models import PredictionResult, TextSubmission

from .forms import LoginForm, RegisterForm
from .models import UserConsent


def register_view(request):
    """
    Handle user registration with consent
    """
    # If user is already logged in, redirect to predictions page
    if request.user.is_authenticated:
        messages.info(request, 'You are already logged in.')
        return redirect('predictions:input')

    if request.method == 'POST':
        form = RegisterForm(request.POST)

        if form.is_valid():
            # Create the user
            user = form.save()

            # Create UserConsent record with consent given
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
            # Form has errors - they will be displayed in the template
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{error}')
    else:
        form = RegisterForm()

    context = {'form': form}
    return render(request, 'accounts/register.html', context)


def login_view(request):
    """
    Handle user login with consent check
    """
    # If user is already logged in, redirect to predictions page
    if request.user.is_authenticated:
        messages.info(request, 'You are already logged in.')
        return redirect('predictions:input')

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)

        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')

            # Authenticate user
            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                if user.is_staff:
                    messages.success(request, f'Welcome back, {username}!')
                    return redirect('/')

                # Check if user has consented
                try:
                    consent = UserConsent.objects.get(user=user)
                    if not consent.has_consented:
                        messages.warning(
                            request,
                            'Please review and accept our data processing terms to continue.',
                        )
                        return redirect('accounts:consent')
                except UserConsent.DoesNotExist:
                    # No consent record - create one and redirect to consent page
                    UserConsent.objects.create(user=user, has_consented=False)
                    messages.warning(
                        request,
                        'Please review and accept our data processing terms to continue.',
                    )
                    return redirect('accounts:consent')

                messages.success(request, f'Welcome back, {username}!')

                # Redirect to next page if specified, otherwise to predictions
                next_page = request.GET.get('next', 'predictions:input')
                return redirect(next_page)
        else:
            messages.error(request, 'Invalid username or password. Please try again.')
    else:
        form = LoginForm()

    context = {'form': form}
    return render(request, 'accounts/login.html', context)


@login_required(login_url='accounts:login')
def logout_view(request):
    """
    Handle user logout and end session
    """
    username = request.user.username
    logout(request)
    messages.success(
        request, f'You have been logged out successfully. See you soon, {username}!'
    )
    return redirect('accounts:login')


@login_required(login_url='accounts:login')
def consent_view(request):
    """
    Handle data processing consent page
    """
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
            messages.error(request, 'You must consent to continue using the service.')

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
    # Get total number of analyses for this user
    total_analyses = TextSubmission.objects.filter(user=request.user).count()

    # Get consent status
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


@login_required(login_url='accounts:login')
@require_http_methods(['POST'])
def change_password_api(request):
    """
    API endpoint to change user password
    """
    try:
        # Parse JSON body
        data = json.loads(request.body)
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')

        # Validate inputs
        if not current_password or not new_password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'missing_fields',
                    'message': 'Both current and new password are required',
                },
                status=400,
            )

        # Check if current password is correct
        if not request.user.check_password(current_password):
            return JsonResponse(
                {
                    'success': False,
                    'error': 'incorrect_password',
                    'message': 'Current password is incorrect',
                },
                status=400,
            )

        # Check if new password is different
        if current_password == new_password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'same_password',
                    'message': 'New password must be different from current password',
                },
                status=400,
            )

        # Validate new password strength
        if len(new_password) < 8:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'weak_password',
                    'message': 'Password must be at least 8 characters long',
                },
                status=400,
            )

        # Set new password (automatically hashes it)
        request.user.set_password(new_password)
        request.user.save()

        # Important: Update session to prevent logout
        update_session_auth_hash(request, request.user)

        return JsonResponse(
            {'success': True, 'message': 'Password changed successfully'}
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {'success': False, 'error': 'invalid_json', 'message': 'Invalid JSON data'},
            status=400,
        )
    except Exception as e:
        return JsonResponse(
            {'success': False, 'error': 'server_error', 'message': str(e)}, status=500
        )


@login_required(login_url='accounts:login')
@require_http_methods(['DELETE'])
def delete_all_data_api(request):
    """
    API endpoint to delete all user's analysis data (but keep account)
    Also revokes consent so user needs to re-consent
    """
    try:
        # Parse JSON body
        data = json.loads(request.body)
        password = data.get('password')

        # Validate password
        if not password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'missing_password',
                    'message': 'Password is required',
                },
                status=400,
            )

        # Check if password is correct
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
        deleted_count = deletion_result[0]  # Total number of objects deleted

        # Revoke consent - user will need to re-consent
        try:
            consent = UserConsent.objects.get(user=request.user)
            consent.revoke_consent()
        except UserConsent.DoesNotExist:
            # Create a revoked consent record
            UserConsent.objects.create(
                user=request.user,
                has_consented=False,
                revoked_at=timezone.now(),
            )

        # Determine redirect URL based on user type
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
    except Exception as e:
        return JsonResponse(
            {'success': False, 'error': 'server_error', 'message': str(e)}, status=500
        )


@login_required(login_url='accounts:login')
@require_http_methods(['DELETE'])
def delete_account_api(request):
    """
    API endpoint to permanently delete user account and all associated data

    """
    try:
        # Parse JSON body
        data = json.loads(request.body)
        password = data.get('password')

        # Validate password
        if not password:
            return JsonResponse(
                {
                    'success': False,
                    'error': 'missing_password',
                    'message': 'Password is required',
                },
                status=400,
            )

        # Check if password is correct
        if not request.user.check_password(password):
            return JsonResponse(
                {
                    'success': False,
                    'error': 'incorrect_password',
                    'message': 'Incorrect password',
                },
                status=400,
            )

        # Store username for response
        username = request.user.username

        # Delete user account (this will cascade delete all related data)
        # including all TextSubmission and PredictionResult records due to ForeignKey
        request.user.delete()

        # Logout (session is destroyed)
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
    except Exception as e:
        return JsonResponse(
            {'success': False, 'error': 'server_error', 'message': str(e)}, status=500
        )


@login_required(login_url='accounts:login')
def history_view(request):
    """
    Display user's analysis history with charts and statistics.

    This view fetches all prediction results for the logged-in user from the database
    and calculates statistics for display on the history page.

    Note: mental_state values are normalized to lowercase for consistent comparison,
    as the database may store them with varying capitalization (e.g., 'Stress' vs 'stress').
    """
    # Get all submissions for this user with related prediction results
    # Using select_related for efficient database query (single JOIN instead of N+1)
    submissions = (
        TextSubmission.objects.filter(user=request.user)
        .select_related('predictionresult')
        .order_by('-submitted_at')
    )

    # Build list of analyses with all necessary data for the template
    # IMPORTANT: Normalize mental_state to lowercase for consistent filtering and counting
    analyses = []
    for submission in submissions:
        try:
            prediction = submission.predictionresult
            # Normalize mental_state to lowercase for consistent comparison
            raw_mental_state = prediction.mental_state or ''
            normalized_state = raw_mental_state.lower().strip()

            analyses.append(
                {
                    'id': submission.id,
                    'text': submission.text_content,
                    'mental_state': normalized_state,  # Use normalized lowercase value
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
            # Submission without a prediction result - skip it
            continue

    # Calculate statistics using normalized (lowercase) mental_state values
    total_analyses = len(analyses)
    normal_count = sum(1 for a in analyses if a['mental_state'] == 'normal')
    concern_count = total_analyses - normal_count

    # Get last analysis date
    last_analysis = 'Never'
    if analyses:
        last_analysis = analyses[0]['created_at'].strftime('%b %d, %Y')

    # Calculate state distribution with percentages
    # mental_state is already normalized to lowercase in the analyses list
    state_counts = defaultdict(int)
    for analysis in analyses:
        state_counts[analysis['mental_state']] += 1

    # Build state distribution dictionary with counts and percentages
    # Using a simple structure that works reliably with Django templates
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

    # Pagination - 10 items per page
    paginator = Paginator(analyses, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'total_analyses': total_analyses,
        'normal_count': normal_count,
        'concern_count': concern_count,
        'last_analysis': last_analysis,
        'state_distribution': state_distribution,
        'analyses': page_obj,  # Paginated analyses
        'page_obj': page_obj,
        'is_paginated': paginator.num_pages > 1,
    }

    return render(request, 'accounts/history.html', context)


@login_required(login_url='accounts:login')
@require_http_methods(['GET'])
def chart_data_api(request):
    """
    API endpoint to fetch chart data for mental health trends over time.

    Query parameters:
        - period: 'week', 'month', or 'all' (default: 'week')

    Returns JSON with labels and datasets for Chart.js
    """
    period = request.GET.get('period', 'week')
    now = timezone.now()

    # Determine date range based on period
    # More date labels for better visualization
    if period == 'week':
        start_date = now - timedelta(days=7)
        # Generate daily labels for the past week (8 labels)
        labels = [(start_date + timedelta(days=i)).strftime('%a %d') for i in range(8)]
        group_days = 1
    elif period == 'month':
        start_date = now - timedelta(days=30)
        # Generate labels for every 2 days (15 labels for better granularity)
        labels = [
            (start_date + timedelta(days=i * 2)).strftime('%b %d') for i in range(16)
        ]
        group_days = 2
    else:  # 'all' - show last 90 days
        start_date = now - timedelta(days=90)
        # Generate labels for every 7 days (13 labels - about 3 months of weekly data)
        labels = [
            (start_date + timedelta(days=i * 7)).strftime('%b %d') for i in range(14)
        ]
        group_days = 7

    # Query predictions within the date range for the current user
    submissions = (
        TextSubmission.objects.filter(
            user=request.user, submitted_at__gte=start_date, submitted_at__lte=now
        )
        .select_related('predictionresult')
        .order_by('submitted_at')
    )

    # Initialize data structure for each mental state
    mental_states = ['normal', 'depression', 'stress', 'suicidal']
    chart_data = {state: [0] * len(labels) for state in mental_states}

    # Count occurrences for each mental state over time
    for submission in submissions:
        try:
            prediction = submission.predictionresult
            # Normalize mental_state to lowercase for consistent comparison
            raw_state = prediction.mental_state or ''
            state = raw_state.lower().strip()

            if state not in mental_states:
                continue

            # Determine which bucket this prediction falls into
            days_diff = (submission.submitted_at.date() - start_date.date()).days
            index = min(max(0, days_diff // group_days), len(labels) - 1)

            chart_data[state][index] += 1

        except PredictionResult.DoesNotExist:
            continue

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

    return JsonResponse(
        {
            'labels': labels,
            'datasets': datasets,
        }
    )


@login_required(login_url='accounts:login')
@require_http_methods(['DELETE'])
def delete_analysis_api(request, analysis_id):
    """
    API endpoint to delete a single analysis.

    This allows users to delete individual analyses from their history.
    The analysis must belong to the current user.

    Args:
        request: The HTTP request
        analysis_id: The ID of the PredictionResult to delete

    Returns:
        JSON response with success status and message
    """
    try:
        # Find the prediction result and verify it belongs to the current user
        prediction = PredictionResult.objects.select_related('submission').get(
            id=analysis_id, submission__user=request.user
        )

        # Get the associated submission
        submission = prediction.submission

        # Delete the prediction (this will also cascade if set up that way)
        prediction.delete()

        # Delete the submission
        submission.delete()

        return JsonResponse(
            {'success': True, 'message': 'Analysis deleted successfully'}
        )

    except PredictionResult.DoesNotExist:
        return JsonResponse(
            {'success': False, 'error': 'Analysis not found or access denied'},
            status=404,
        )

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required(login_url='accounts:login')
@require_http_methods(['GET'])
def export_data_api(request):
    """
    API endpoint to export all user data in JSON format
    For GDPR compliance and user data portability
    """
    user = request.user

    try:
        # Export user profile data
        user_data = {
            'username': user.username,
            'email': user.email,
            'date_joined': user.date_joined.isoformat(),
        }

        # Export consent data
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

        # Export analysis history
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

    except Exception as e:
        return JsonResponse(
            {'success': False, 'error': 'server_error', 'message': str(e)}, status=500
        )
