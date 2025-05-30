from django.urls import path
from . import views

urlpatterns = [
    path('dataset/', views.dataset, name='dataset'),                  # Navbar 1: index.html
    path('dataset/', views.dataset, name='dataset'),    # Navbar 2: dataset.html
    path('testing/', views.testing, name='testing'),    # Navbar 3: testing.html
    path('klasifikasi/', views.klasifikasi, name='klasifikasi'),  # Navbar 4: klasifikasi.html
]
