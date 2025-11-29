from django.shortcuts import redirect
from django.urls import reverse

class ConsentMiddleware:
    """Check if the user has consented to data processing"""

    def __init__(self, get_response):
        self.get_response = get_response
        # Paths that do not require consent check
        self.exempt_paths = [
            '/accounts/login/',
            '/accounts/register/',
            '/accounts/logout/',
            '/accounts/consent/',
            '/accounts/privacy/',
            '/static/',
            '/media/',
            '/',  # Home page
        ]

    def __call__(self, request):
        # Check if the path requires consent
        if request.user.is_authenticated:
            path = request.path

            # Check if the path is exempt
            is_exempt = any(path.startswith(exempt) for exempt in self.exempt_paths)

            if not is_exempt:
                # Check if the user has consented
                try:
                    consent = request.user.data_consent
                    if not consent.has_consented:
                        return redirect('accounts:consent')
                except:
                    # No consent record yet, redirect to consent page
                    return redirect('accounts:consent')

        response = self.get_response(request)
        return response