import ipywidgets as widgets
import os
from ipywidgets import interactive

""""Generacion del las funciones para crear los widgets de seleccion de parametros para la solucion 1"""
def widget_CIE():
    mi = widgets.IntSlider(
        value=120,
        min=0,
        max=255,
        step=1,
        description='Min thresh:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )
    
    ma = widgets.IntSlider(
        value=150,
        min=0,
        max=255,
        step=1,
        description='Max thresh:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )

    a = widgets.IntSlider(
        value=125,
        min=0,
        max=255,
        step=1,
        description='a:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )

    b = widgets.IntSlider(
        value=140,
        min=0,
        max=255,
        step=1,
        description='b:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )

    return [mi, ma, a, b]

def widget_RGB():
    intensity_slider = widgets.FloatSlider(
        value=1.0,
        min=0,
        max=10.0,
        step=0.01,
        description='Intensity:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )
    min_brightness_slider = widgets.FloatSlider(
        value=0.0,
        min=0,
        max=10.0,
        step=0.01,
        description='Minimum Mask Brightness:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )
    min_ratio_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=10.0,
        step=0.01,
        description='Ratio Red/Blue Min:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )
    max_ratio_slider = widgets.FloatSlider(
        value=0.33,
        min=0,
        max=10.0,
        step=0.01,
        description='Ratio Red/Blue Max:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )
    radius_slider = widgets.FloatSlider(
        value=5.0,
        min=0,
        max=10.0,
        step=0.01,
        description='Radius:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    return [intensity_slider, min_brightness_slider, min_ratio_slider, max_ratio_slider, radius_slider]

def img_slid(path_img):
    
    # cargamos todas las imagenes que tenemos
    image_list = []
    counter = 1
    for file in os.listdir(path_img):
      image_list.append((file,counter))
      counter += 1

    img = widgets.Dropdown(
        options=image_list,
        value=5,
        description='Imagen:',
    )
    return img
