import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure 


def plot(img, name, space):
  """Creamos una grafica donde cada banda de la imagen es visualizada separadamente
      de esta manera podemos ver que bandas capturan mejor la aberracion
      img: imagen 
      name:nombre del plot
      space: espacio de color que queremos usar para pintar, toma 2 valores posibles 'LAB' o 'RGB' """
  
  fig = plt.figure(figsize=(12,3))
  fig.suptitle(name, fontsize=16)
  ax = fig.add_subplot(1, 4, 1)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  ax.imshow(img_rgb)
  ax.set_xlabel(name,fontsize=14)
  if space == 'RGB':
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color_legend = ['R: Rojo','G: Verde', 'B: Azul']
  elif space == 'LAB':
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    color_legend = ['L: Luminosidad','A: Verde-Rojo', 'B: Azul-Amarillo']
  for idx in range(img.shape[2]):
    ax = fig.add_subplot(1, 4, idx+2) 
    ax.imshow(img[:,:,idx]) 
    ax.set_xlabel(color_legend[idx],fontsize=14)
    
  #scatter plot
  for idx, col in enumerate(color):
    histr = cv2.calcHist([img],[idx],None,[256],[0,256])
    plt.plot(histr, color = color_legend, label = name[idx])
    plt.xlim([0,256])
  leg = plt.legend(loc='best')
  for l in leg.legendHandles:
    l.set_linewidth(10)
  plt.show()

def plot_scatter(img, name, color, mask = None):
  """Creamos una grafica donde cada banda de la imagen es representada en forma de histograma
      esto nos ayuda a ver la distribucion de los valores
      img: imagen 
      name:nombre del plot
      color: lista con los colores de cada subplot"""
  fig = plt.figure(figsize=(12,3))
  for idx, col in enumerate(color):
    histr = cv2.calcHist([img],[idx],None,[256],[0,256])
    plt.plot(histr, color = col, label = name[idx])
    plt.xlim([0,256])
  leg = plt.legend(loc='best')
  for l in leg.legendHandles:
    l.set_linewidth(10)
  plt.title(name)
  plt.show()

def moving_w(k, img, mask, funct):
  """Metodo donde dando un kernel (matriz), movimiento 
      y una region se realize la operacion deseada"""
  idx = np.where(mask == 1)
  iter = idx[1].shape[0]
  print("Total de iteraciones de la MV: {}".format(iter))
  #creacion de la ventana
  margen = int(k/2)
  cols = [None,None]
  fils = [None,None]

  for i in range(iter):
    #control de bordes
    if (idx[0][i] - margen) >= 0:
      cols[0] = idx[0][i] - margen
    else:
      cols[0] = 0
    if (idx[0][i] + margen) < img.shape[0]:
      cols[1] = idx[0][i] + margen
    else:
      cols[1] = img.shape[0] -1
    if (idx[1][i] - margen) >= 0:
      fils[0] = idx[1][i] - margen
    else:
      fils[0] = 0
    if (idx[1][i] + margen) < img.shape[1]:
      fils[1] = idx[1][i] + margen
    else:
      fils[1] = img.shape[1] -1
    
    subimage = img[cols[0]:cols[1], fils[0]:fils[1],:]
    ########AQUI se APLICARIA EL FILTRO
    result = funct(subimage)
    ###################################
    if result[1]:
      img[cols[0]:cols[1], fils[0]:fils[1], :] = result[0]
    else:
      img[idx[0][i], idx[1][i], 0] = result[0][0]
      img[idx[0][i], idx[1][i], 1] = result[0][1]
      img[idx[0][i], idx[1][i], 2] = result[0][2]
  return img

def hist_thresh(img, banda):
  """Codigo para extraer de manera automatica un valor threshold a partir de 
  maximizar la variancia entre los grupos o bins del histograma.
  Codigo basado en: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
  img: imagen a analizar
  banda: banda donde se hara la binarizacion
  Se retorna la mascara binaria
  """
  imgh = img[:,:,banda]
  bins_num = np.max(imgh)
  bins_num = 255
  # Obtenemos histograma de la imagen
  hist, bin_edges = np.histogram(imgh, bins=bins_num)
  # Normalizamos el histograma
  hist = np.divide(hist.ravel(), hist.max())
  # Calculamos el centro de los bins dels histograma
  bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
  # iteramos sobre hist para obtener las probablidades y calculamos sus medias correspondientes
  weight1 = np.cumsum(hist)
  weight2 = np.cumsum(hist[::-1])[::-1]
  mean1 = np.cumsum(hist * bin_mids) / weight1
  mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

  inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
  # Maximizamos la varianzia entre clases escogiendo el maximo valor mas repetido
  index_of_max_val = np.argmax(inter_class_variance)
  thresh = bin_mids[:-1][index_of_max_val]
  print("Los valores escogidos para el thresh, en la banda {} son:{}, {}".format(banda, np.max(img[:,:,banda]) , thresh))
  
  #Realizamos el threshold con el valor maximo y el obtenido en el metodo anterior
  mask_min = cv2.threshold(img[:,:,banda], thresh, np.max(img[:,:,banda]), cv2.THRESH_BINARY)
  mask_min = np.array(mask_min[1])
  #Creamos la mascara binaria
  bin_mask_min = np.where(mask_min > 0, 1, 0)
  return bin_mask_min

def autocontraste(x,a,b,minv=0,maxv=255):
    '''Realiza el autocontraste llevando los puntos a y b a 0 y 225, respectivamente'''
    if a < minv:
        a = minv
    if a > maxv:
        a = maxv
    if b < minv:
        b = minv
    if b > maxv:
        b = maxv
        
    if x<a:
        y=minv
    elif x>b:
        y=maxv
    else:
        m = maxv-minv
        n = b*minv - a*maxv
        y=round((m*x+n)/(b-a))
    return y



def median(img):
  x = np.median(img[:,:,0])
  y = np.median(img[:,:,1])
  z = np.median(img[:,:,2])
  return [[x,y,z], False]

#definiciones de funciones usadas para la solucion 3
class Params:
    radius = 5.0
    intensity = 1.0
    min_brightness = 0.0
    min_red_to_blue_ratio = 0.0
    max_red_to_blue_ratio = 0.33
   
#perform motion blur on a column in an image
def MotionBlurX(radius, w, h, img):
    # make new array to hold the blur and initialize to full black
    blur = np.full_like(img,0)
    
    #compute blur range
    ww = 2 * radius + 1
    for i in range(w):
        # compute initial acc
        acc = (radius + 2) * img[i,0,2]
        for j in range(1,int(radius)):
            acc += img[i,j,2]

        #perform blur operation
        for j in range(h):
            acc = acc - img[i,max(0,j-1-int(radius)),2] + img[i,min(h-1, j+int(radius)), 2]
            blur[i,j,2]=acc/ww

    #done -> return blur
    return blur

def MotionBlurY(radius, w, h, img):
    # make new array to hold the blur and initialize to full black
    blur = np.full_like(img,0)
    
    #compute blur range
    ww = 2 * radius + 1
    for j in range(h):
        #compute initial acc
        acc = (radius + 2)*img[0,j,2]
        for i in range(1,int(radius)):
            acc += img[i,j,2]

        #perform blur operation
        for i in range(w):
            acc = acc - img[max(0,i-1-int(radius)),j,2] + img[min(w-1,i+int(radius)),j ,2];
            blur[i,j,2]=acc/ww
    
    #done -> return blur
    return blur

def BoxBlur(radius,w,h,img):
    blurX = MotionBlurX(radius,w,h,img)
    blurY = MotionBlurY(radius,w,h,blurX)
    return blurY

def DivUp(a,b):
    r = a/b
    if a%b == 0:
        return r
    else:
        return (r+1)

def TentBlur(radius, w,h, img):
    tmp = BoxBlur(DivUp(radius,2),w,h,img)
    blur = BoxBlur(DivUp(radius,2),w,h,tmp)
    return tmp

def MakePurpleBlur(params, img_original):
    
    #get img dimensions
    dims = img_original.shape

    # make new array to hold the blur and initialize to full black
    mask = np.full_like(img_original,0)
    
    #store min brightness
    thresh = params.min_brightness

    #use the blue component to define the intensity of the light
    #subject to unfocusing
    for i in range(dims[0]):
        for j in range(dims[1]):
            b = img_original[i,j,2]/255.0
            grey = max(0.0, (b-thresh))*(1.0/(1.0-thresh))
            p = params.intensity*grey
            mask[i,j,2] = p*255

    blur = TentBlur(params.radius,dims[0],dims[1],mask)
    return blur

def RemovePurpleBlur(params, img, mask):
    #copy img to result
    res = img.copy()
    #get img dimensions
    dims = img.shape
    
    for i in range(dims[0]):
        for j in range(dims[1]):
            imgColor = img[i,j,:] / 255.0
            mskColor = mask[i,j,:] / 255.0
            bl = min(1, mskColor[2])

            #amount of blue and red that would produce a grey if removed
            db = max(0, imgColor[2] - imgColor[1])
            dr = max(0, imgColor[0] - imgColor[1])

            #maximum amount of blue that we accept to remove, ignoring red level
            mb = min(bl,db)

            #amount of red that we will remove, honoring max red:blue ratio
            r_diff = min(dr, params.max_red_to_blue_ratio)

            #amount of blue that we will remove, honoring min red:blue ratio
            if params.min_red_to_blue_ratio > 0:
                b_diff = min(mb, r_diff / params.min_red_to_blue_ratio)
            else:
                b_diff = mb

            res[i,j,0] = (imgColor[0]-r_diff) * 255.0
            res[i,j,1] = (imgColor[1]       ) * 255.0
            res[i,j,2] = (imgColor[2]-b_diff) * 255.0

    #done return res
    return res

def Unpurple(params, img_original):
    #generate blurred mask from original image
    mask = MakePurpleBlur(params, img_original)

    #use blurred mask to remove fringing
    output = RemovePurpleBlur(params, img_original, mask)

    return mask,output
  
 #Estas funciones no se usan en el codigo pero han sido intentos realizados durante la practica
def im_resize(img, alpha):
  width = int(img.shape[1] * alpha)
  height = int(img.shape[0] * alpha)
  dim = (width, height)
  im2 = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
  return im2

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r

def scaleim(img, alpha):
  ydim,xdim = img.shape
  if (alpha > 1):
    im2 = im_resize(img, alpha)
    ydim2,xdim2 = im2.shape
    cy = int(np.floor((ydim2-ydim)/2))
    cx = int(np.floor((xdim2-xdim)/2))
    im2 = im2[cy:(ydim2-cy), cx:(xdim2-cx)]
    im2 = cv2.resize(im2, (xdim, ydim), interpolation = cv2.INTER_CUBIC)

  else:
    im2 = im_resize(img, alpha)
    ydim2,xdim2 = im2.shape
    im3 = np.zeros((ydim, xdim))
    cy = int(np.floor((ydim-ydim2)/2))
    cx = int(np.floor((xdim-xdim2)/2))
    im3[cy:(ydim2+cy), cx:(xdim2+cx)] = im2
    idx = np.where(im3 == 0)
    for i in range(len(idx[0])):
      im3[idx[0][i], idx[1][i]] = img[idx[0][i], idx[1][i]]
    im2 = im3
  return im2

def generar_franja(img, alphaR, alphaB):
  y,x,z = img.shape
  copyim =img.copy()
  copyim[:,:,0] = scaleim(copyim[:,:,0], alphaB)
  copyim[:,:,2] = scaleim(copyim[:,:,2], alphaR)
  return copyim

def corregir_franja(copyimg):
  img =copyimg.copy()
  k = 0
  green = img[:,:,1]
  C = np.empty((len(np.arange(0.8, 1.21, 0.02)),3))

  for alpha in np.arange(0.8, 1.21, 0.02):
    alpha = round(alpha,2)
    red = scaleim(img[:,:,2], 1/alpha)
    ind = np.where(red > 0)
    C[k][0] = corr2(red[ind[0][0]:ind[0][-1], ind[1][0]:ind[1][-1]], green[ind[0][0]:ind[0][-1], ind[1][0]:ind[1][-1]])
    blue = scaleim(img[:,:,0], 1/alpha)
    ind = np.where(blue > 0)
    C[k][1] = corr2(blue[ind[0][0]:ind[0][-1], ind[1][0]:ind[1][-1]], green[ind[0][0]:ind[0][-1], ind[1][0]:ind[1][-1]])
    C[k][2] = alpha
    k += 1
  maxval = np.max(C[:,0])
  maxindR = np.argmax(C[:,0])
  maxval = np.max(C[:,1])
  maxindB = np.argmax(C[:,1])
  alphaR = C[maxindR,2]
  alphaB = C[maxindB,2]
  print("Las alphas de aberracion de esta images son rojo: {}, azul:{}".format(alphaR, alphaB))
  imCORRECT = img
  imCORRECT[:,:,2] = scaleim(img[:,:,2], 1/alphaR)
  imCORRECT[:,:,0] = scaleim(img[:,:,0], 1/alphaB)
  return imCORRECT

def realce(x,A,B):
    ''' Función para realizar el realce: si A < B = realce sombras y si A > B = realce claros'''
    if A < 0:
        A=0
    if A > 255:
        A=255
    if B < 0:
        B=0
    if B > 255:
        B=255
    if x <= A:
        y = round(B*x/A)
    else:
        y = round(((255-B)*x+(255*(B-A)))/(255-A))
    return y

def logaritmo(x, maxv, alfa=0.5):
    c = maxv/np.log(1+(np.e**alfa-1)*maxv)
    y=c*np.log(1+(np.e**alfa-1)*x)
    return round(y)

def exponencial(x, maxv, alfa=10):
    x=x/maxv
    c = maxv/((1+alfa)-1)
    y=c*((1+alfa)**x-1)
    return y
