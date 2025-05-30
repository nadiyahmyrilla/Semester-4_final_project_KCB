from django.urls import path
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),                  # Navbar 1: index.html
    path('dataset/', views.dataset, name='dataset'),    # Navbar 2: dataset.html
    path('testing/', views.testing, name='testing'),    # Navbar 3: testing.html
    path('klasifikasi/', views.klasifikasi, name='klasifikasi'),  # Navbar 4: klasifikasi.html
]
