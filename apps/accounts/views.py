from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("prediction:input")
        else:
            context ={"error": "Invalid username of password"}
            return render(request, "accounts/login.html", context)

    return render (request, "accounts/login.html")



def logout_view(request):
    return render(request, 'home')


def register_view(request):
    return render(request, 'accounts/register.html', {})

def history_view(request):
    return render(request, 'accounts/history.html')
