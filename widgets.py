import ipywidgets as widgets
import os
from ipywidgets import interactive

""""Generacion del las funciones para crear los widgets de seleccion de parametros para la solucion 1"""
def widget_CIE(path_img):
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

    # cargamos todas las imagenes que tenemos
    image_list = []
    counter = 1
    for file in os.listdir(path_img):
      image_list.append((file,counter))
      counter += 1

    img = widgets.Dropdown(
        options=image_list,
        value=2,
        description='Imagen:',
    )
    return [mi, ma, a, b, img, image_list]
