from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

from uploads.core import views


urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
    url(r'^demo/cntk/$', views.demo_cntk, name='demo_cntk'),
    url(r'^uploads/process/$', views.process, name='process'),
    url(r'^uploads/form/$', views.model_form_upload, name='model_form_upload'),
    url(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.OUT_URL, document_root=settings.OUT_ROOT)
    urlpatterns += static(settings.OUTCNTK_URL, document_root=settings.OUTCNTK_ROOT)
