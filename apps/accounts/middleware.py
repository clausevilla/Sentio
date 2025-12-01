# Author: Lian Shi

# Disclaimer: LLM has been used to suggest consent middleware setup, and implementation was made to ensure correctness and fit with project requirements.

from django.shortcuts import redirect


class ConsentMiddleware:
    """
    Middleware to check if authenticated users have given consent.
    Redirects to consent page if consent is not given or has been revoked.
    """

    # Paths that don't require consent check
    EXEMPT_PATHS = [
        '/accounts/login/',
        '/accounts/logout/',
        '/accounts/register/',
        '/accounts/consent/',
        '/accounts/privacy/',
        '/static/',
        '/media/',
        '/admin/',
        '/ml-admin/',
        '/',
    ]

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip consent check for exempt paths
        path = request.path
        for exempt in self.EXEMPT_PATHS:
            if path.startswith(exempt) or path == exempt:
                return self.get_response(request)

        # Only check consent for authenticated users
        if request.user.is_authenticated:
            try:
                from .models import UserConsent

                consent = UserConsent.objects.get(user=request.user)
                if not consent.has_consented:
                    # User hasn't consented or consent was revoked
                    return redirect('accounts:consent')
            except UserConsent.DoesNotExist:
                # No consent record exists - redirect to consent page
                return redirect('accounts:consent')
            except Exception:
                # If there's any error, let the request through
                # (fail open to avoid blocking users)
                pass

        return self.get_response(request)
