# Author: Lian Shi
# Disclaimer: LLM has been used to debug consent middleware implementation while it is not working properly to check user consent status and redirect to consent page accordingly. Manual adjustment was done to fix the issues.

from django.shortcuts import redirect


class ConsentMiddleware:
    """
    Middleware to check if authenticated users have given consent.
    Redirects to consent page if consent is not given or has been revoked.
    """

    # Paths that don't require consent check (use startswith matching)
    EXEMPT_PATH_PREFIXES = [
        '/accounts/login',
        '/accounts/logout',
        '/accounts/register',
        '/accounts/consent',
        '/accounts/privacy',
        '/about/',
        '/static/',
        '/media/',
        '/admin/',
        '/management/',
    ]

    # Exact paths that are exempt
    EXEMPT_EXACT_PATHS = [
        '/',
    ]

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path

        # Skip consent check for exact exempt paths
        if path in self.EXEMPT_EXACT_PATHS:
            return self.get_response(request)

        # Skip consent check for paths that start with exempt prefixes
        for prefix in self.EXEMPT_PATH_PREFIXES:
            if path.startswith(prefix):
                return self.get_response(request)

        # Only check consent for authenticated users
        if request.user.is_authenticated:
            # Exempt staff and superusers from consent check
            if request.user.is_staff or request.user.is_superuser:
                return self.get_response(request)
            try:
                from .models import UserConsent

                consent = UserConsent.objects.get(user=request.user)
                if not consent.has_consented:
                    # User hasn't consented or consent was revoked
                    return redirect('accounts:consent')
            except UserConsent.DoesNotExist:
                # No consent record exists - redirect to consent page
                return redirect('accounts:consent')
            except Exception as e:
                # Log the error but let the request through to avoid blocking users
                print(f'[ConsentMiddleware] Error checking consent: {e}')
                pass

        return self.get_response(request)
