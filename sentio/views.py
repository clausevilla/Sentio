from django.shortcuts import redirect, render
from django.templatetags.static import static


def favicon_redirect(request):
    return redirect(static('images/favicon.svg'), permanent=True)


def home(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')
