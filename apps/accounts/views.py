# Author: Lian Shi
# Disclaimer: LLM has been used to help generate changepassword and delete account API endpoints.
# Updated: Added consent management functionality

import json
from datetime import datetime, timedelta

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

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
                user=user,
                has_consented=True,
                consent_at=timezone.now()
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

                # Check if user has consented
                try:
                    consent = UserConsent.objects.get(user=user)
                    if not consent.has_consented:
                        messages.warning(
                            request,
                            'Please review and accept our data processing terms to continue.'
                        )
                        return redirect('accounts:consent')
                except UserConsent.DoesNotExist:
                    # No consent record - create one and redirect to consent page
                    UserConsent.objects.create(user=user, has_consented=False)
                    messages.warning(
                        request,
                        'Please review and accept our data processing terms to continue.'
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

            messages.success(request, 'Thank you for consenting to our data processing terms.')
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
    total_analyses = 0
    # TODO: Uncomment when MentalHealthAnalysis model is ready
    # from apps.predictions.models import MentalHealthAnalysis

    # Get total number of analyses for this user
    # For now, using placeholder value
    # total_analyses = MentalHealthAnalysis.objects.filter(user=request.user).count()

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
@require_http_methods(['POST'])
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

        # Delete all user's mental health analyses
        # TODO: Uncomment when MentalHealthAnalysis model is ready
        # from apps.predictions.models import MentalHealthAnalysis
        # deleted_count = MentalHealthAnalysis.objects.filter(user=request.user).delete()[0]
        deleted_count = 0

        # Revoke consent - user will need to re-consent
        try:
            consent = UserConsent.objects.get(user=request.user)
            consent.revoke_consent()
        except UserConsent.DoesNotExist:
            # Create a revoked consent record
            UserConsent.objects.create(
                user=request.user,
                has_consented=False,
                consent_revoked_at=timezone.now()
            )

        return JsonResponse(
            {
                'success': True,
                'message': f'Successfully deleted {deleted_count} analyses. Your consent has been revoked.',
                'deleted_count': deleted_count,
                'consent_revoked': True,
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
@require_http_methods(['POST'])
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
        # including all MentalHealthAnalysis records due to ForeignKey
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
    Display user's analysis history with charts and statistics
    """
    # TODO: Import the MentalHealthAnalysis model when it's ready
    # from apps.predictions.models import MentalHealthAnalysis

    # Get all analyses for this user
    # analyses = MentalHealthAnalysis.objects.filter(user=request.user).order_by('-created_at')

    # Calculate statistics
    # For now, using placeholder values
    total_analyses = 0
    normal_count = 0
    concern_count = 0
    last_analysis = 'Never'

    # TODO: Uncomment when model is ready
    # total_analyses = analyses.count()
    # normal_count = analyses.filter(mental_state='normal').count()
    # concern_count = total_analyses - normal_count
    #
    # if analyses.exists():
    #     last_analysis = analyses.first().created_at.strftime('%b %d, %Y')

    # Calculate state distribution
    state_distribution = {
        'normal': {'label': 'Normal', 'icon': '😊', 'class': 'normal', 'count': 0, 'percentage': 0},
        'depression': {'label': 'Depression', 'icon': '😢', 'class': 'depression', 'count': 0, 'percentage': 0},
        'anxiety': {'label': 'Anxiety', 'icon': '😰', 'class': 'anxiety', 'count': 0, 'percentage': 0},
        'stress': {'label': 'Stress', 'icon': '😫', 'class': 'stress', 'count': 0, 'percentage': 0},
        'suicidal': {'label': 'Suicidal', 'icon': '🆘', 'class': 'suicidal', 'count': 0, 'percentage': 0},
        'bipolar': {'label': 'Bipolar', 'icon': '🔄', 'class': 'bipolar', 'count': 0, 'percentage': 0},
    }

    context = {
        'total_analyses': total_analyses,
        'normal_count': normal_count,
        'concern_count': concern_count,
        'last_analysis': last_analysis,
        'state_distribution': state_distribution,
        'analyses': [],
    }

    return render(request, 'accounts/history.html', context)


def get_chart_data(user, period='week'):
    """
    Get chart data for mental health trends over time
    Helper function for history view
    """
    # TODO: Import the MentalHealthAnalysis model when it's ready
    # from apps.predictions.models import MentalHealthAnalysis

    now = datetime.now()

    # Determine date range and labels based on period
    if period == 'week':
        start_date = now - timedelta(days=7)
        labels = [(start_date + timedelta(days=i)).strftime('%a') for i in range(7)]
    elif period == 'month':
        start_date = now - timedelta(days=30)
        labels = [(start_date + timedelta(days=i * 5)).strftime('%b %d') for i in range(6)]
    else:
        start_date = now - timedelta(days=90)
        labels = [(start_date + timedelta(days=i * 15)).strftime('%b %d') for i in range(6)]

    chart_data = {
        'labels': labels,
        'normal': [0] * len(labels),
        'depression': [0] * len(labels),
        'anxiety': [0] * len(labels),
        'stress': [0] * len(labels),
        'suicidal': [0] * len(labels),
        'bipolar': [0] * len(labels),
    }

    # TODO: Get analyses from database when model is ready
    # analyses = MentalHealthAnalysis.objects.filter(
    #     user=user,
    #     created_at__gte=start_date,
    #     created_at__lte=now
    # ).order_by('created_at')

    # TODO: Count occurrences for each mental state over time
    # for analysis in analyses:
    #     if period == 'week':
    #         label = analysis.created_at.strftime('%a')
    #     elif period == 'month':
    #         days_diff = (analysis.created_at.date() - start_date.date()).days
    #         index = min(days_diff // 5, len(labels) - 1)
    #         label = labels[index]
    #     else:
    #         days_diff = (analysis.created_at.date() - start_date.date()).days
    #         index = min(days_diff // 15, len(labels) - 1)
    #         label = labels[index]
    #
    #     if label in labels:
    #         idx = labels.index(label)
    #         state = analysis.mental_state
    #         if state in chart_data:
    #             chart_data[state][idx] += 1

    return chart_data