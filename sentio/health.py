# Author: Lian Shi


from django.db import connection
from django.http import JsonResponse


def liveness(request):
    """Liveness probe - Is the app process running?"""
    return JsonResponse({'status': 'alive'}, status=200)


def readiness(request):
    """Readiness probe - Is the app ready for traffic?"""
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
        return JsonResponse({'status': 'ready', 'database': 'connected'}, status=200)
    except Exception as e:
        return JsonResponse({'status': 'not ready', 'error': str(e)}, status=503)
