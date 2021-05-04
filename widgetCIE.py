import ipywidgets as widgets
import os

""""Generacion del las funciones para crear los widgets de seleccion de parametros para la solucion 1"""
min_slider = widgets.IntSlider(
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
display(min_slider)

max_slider = widgets.IntSlider(
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
display(max_slider)

a_slider = widgets.IntSlider(
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
display(a_slider)

b_slider = widgets.IntSlider(
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
display(b_slider)

# cargamos todas las imagenes que tenemos
image_list = []
counter = 1
for file in os.listdir(path_img):
  image_list.append((file,counter))
  counter += 1

img_dropdown = widgets.Dropdown(
    options=image_list,
    value=2,
    description='Imagen:',
)
display(img_dropdown)
