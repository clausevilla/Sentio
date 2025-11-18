from django.shortcuts import render

# Create your views here.
def input_view(request):
    return render(request, 'predictions/input.html', {})

def result_view(request):
    return render(request, 'predictions/result.html')


