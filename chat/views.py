# chat/views.py
from django.views.decorators.clickjacking import xframe_options_exempt
from django.shortcuts import render

def index(request):
    return render(request, "chat/index.html")

@xframe_options_exempt
def room(request, room_name):
    return render(request, "chat/room.html", {"room_name": room_name})