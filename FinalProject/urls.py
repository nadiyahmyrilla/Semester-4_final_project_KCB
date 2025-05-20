"""
URL configuration for FinalProject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('flower_app.urls')),  # Routing ke aplikasi utama
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Menambahkan static URL agar gambar di dataset_bunga dapat diakses saat debug
if settings.DEBUG:
    urlpatterns += static('/dataset_bunga/', document_root=settings.BASE_DIR / 'dataset_bunga')
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)