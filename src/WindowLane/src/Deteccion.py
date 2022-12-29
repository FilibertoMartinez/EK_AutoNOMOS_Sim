#!/usr/bin/python

#from tkinter import N
#from tokenize import Double
import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


from matplotlib.patches import Polygon
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage


def nothing(x):
    """Hacer nada"""
    pass

def ventanas_deslizantes(img, return_img=False):
    # Numero de ventanas
    n_ventanas = 8
    # Altura por ventana
    altura_ventana = np.int(img.shape[0]//n_ventanas)
    # Ancho de ventanas
    margen = 20
    # Numero minimo de pixeles para recentrar la siguiente ventana
    minimopix = 150
    # Encuentra el pico izquierdo y derecho del histograma
    histogram, pico_izquierda, pico_derecha = find_histogram_peaks(img)

    puntosIZQx=[]
    puntosIZQy=[]
    puntosDERx=[]
    puntosDERy=[]
    # Por cada ventana hacer
    for ventana in range(n_ventanas):
        media_izquierdaX = 0
        media_derechaX = 0
        sumaIzquierda = 0
        sumaDerecha = 0
    
        #Se definen los limites por cada ventana [De los dos lados, izq der]
        ventanaY_abajo = img.shape[0] - (ventana+1)*altura_ventana
        ventanaY_arriba = img.shape[0] - ventana*altura_ventana
        ventanaX_izq_izq = pico_izquierda - margen
        ventanaX_izq_der = pico_izquierda + margen
        ventanaX_der_izq = pico_derecha - margen
        ventanaX_der_der = pico_derecha + margen
        
        #Si dichos limites sobrepasan la imagen, normalizar
        if( ventanaX_izq_izq<0):
            ventanaX_izq_izq = 1
        if(ventanaX_der_der >= img.shape[1]-1):
            ventanaX_der_der =  img.shape[1]-1 
        if(ventanaY_arriba>=img.shape[0]-1):
            ventanaY_arriba = img.shape[0]-1
        if(ventanaY_abajo<=0):
            ventanaY_abajo = 1

        """
        A continuacion:
        Se recorre todo el contenido de cada ventana, primero de
        abajo a arriba y despues de izquierda a derecha
        """
        
        r = ventanaY_abajo
        while(r<ventanaY_arriba):                                       #De abajo a arriba
            #El contador lleva la posicion del pixel dentro de la ventana
            contadorIZQ=ventanaX_izq_izq+1
            contadorDER=ventanaX_der_izq+1
            r+=1                                                        #Se suma el contador de abajo a arriba
            while(contadorIZQ<ventanaX_izq_der):                        #De izquierda a derecha
                posicionIZQ = img[r,contadorIZQ]                        #Valor actual del pixel
                if(posicionIZQ >0):                                     #Si el pixel es blanco
                    media_izquierdaX+=contadorIZQ                       #Suma las posiciones                       
                    sumaIzquierda+=1                                    #Se suma la cantidad de blancos
                    puntosIZQy.append(contadorIZQ)
                    puntosIZQx.append(r)
                contadorIZQ+=1                                          #Se avanza en el eje de [X] de izq a derecha

            while(contadorDER<ventanaX_der_der):                        #LO ANTERIOR SE REPITE AHORA PARA LA PARTE DERECHA
                posicionDER = img[r,contadorDER]
                if(posicionDER >0):
                    media_derechaX += contadorDER 
                    sumaDerecha=sumaDerecha+1
                    puntosDERy.append(contadorDER)
                    puntosDERx.append(r)
                contadorDER+=1

        if(sumaIzquierda>=minimopix):                                   #Si la cantidad de pixeles blancos es mayor que el minimo para recentrar
            media_izquierdaX /=  sumaIzquierda                          #Encuentra la media por cada ventana del lado izquierdo
            pico_izquierda = media_izquierdaX                           #Reasigna el pico izquierda para la siguiente ventana
        if(sumaDerecha>=minimopix):                                     #REPITE PARA LADO DERECHO
            media_derechaX /=  sumaDerecha
            pico_derecha = media_derechaX

        cv2.rectangle(return_img,(ventanaX_izq_izq,ventanaY_abajo),(ventanaX_izq_der,ventanaY_arriba),(0,0,150),2) #Dibuja las ventanas izquierdas
        cv2.rectangle(return_img,(ventanaX_der_izq,ventanaY_abajo),(ventanaX_der_der,ventanaY_arriba),(0,0,150),2) #Dibuja las ventanas derechas

    lineas(puntosIZQx,puntosIZQy,puntosDERx,puntosDERy,img,return_img)  #Ajusta un polinomio a las ventanas

def find_histogram_peaks(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # Peak in the first half indicates the likely position of the left lane
    half_width = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:half_width])

    # Peak in the second half indicates the likely position of the right lane
    rightx_base = np.argmax(histogram[half_width:]) + half_width
    
    return histogram, leftx_base, rightx_base
     
def perspectiva(img, src_coordenadas=None, dst_coordinates=None):
    
    img_size = (img.shape[1], img.shape[0])

    # Se obtiene la transformacion de perspectiva con las coordenadas de ROI y destino
    M = cv2.getPerspectiveTransform(src_coordenadas, dst_coordinates)

    # Se obtiene la transformacion de perspectiva inversa
    Minv = cv2.getPerspectiveTransform(dst_coordinates, src_coordenadas)
    
    # Se crea la perspectiva con interpolacion lineal
    cambioPerspectiva = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return cambioPerspectiva

def difuminacionMedian(image,kernel=5):
    '''
    Funcion para reducir el ruido en imagenes mediante una funcion Mediana [Lineal]
    '''
    difuminacion=cv2.medianBlur(image, kernel)
    return difuminacion

def difuminacionGauss(image, kernel=10):
    '''
    Funcion para reducir el ruido en imagenes mediante una funcion Gaussiana [En espacio X,Y]
    '''
    difuminacion = cv2.GaussianBlur(image, (kernel,kernel), 0)
    return difuminacion

def grises(image):
    grises = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grises

def BlancoNegro(image,Down,Up):
    BlackWhite=cv2.inRange(image, Down, Up);
    return BlackWhite

def ROI(frame,escala):
    frame_size = (frame.shape[1], frame.shape[0])         #Tamano de la imagen [X,Y]
    escala = escala / float(100)
    roi=frame.copy()

    src_coordenadas = np.float32(
        [[1,  405*escala],  # Abajo izquierda
         [150*escala,  310*escala],  # Arriba izquierda
         [(frame_size[0]-150*escala),  310*escala],  # Arriba derecha
         [(frame_size[0]-1), 405*escala]]) # Abajo derecha   
    
    dst_coordenadas = np.float32(
        [[1,  frame_size[1]],  # Abajo izquierda
         [1,  1],  # Arriba izquierda
         [frame_size[0],  1],  # Arriba derecha
         [frame_size[0], frame_size[1]]]) # Abajo derecha

    #DIBUJAR LINEAS DE FUENTE
    cv2.line(roi, (src_coordenadas[0,0],src_coordenadas[0,1]), (src_coordenadas[1,0],src_coordenadas[1,1]), (157,0,255), 3) 
    cv2.line(roi, (src_coordenadas[1,0],src_coordenadas[1,1]), (src_coordenadas[2,0],src_coordenadas[2,1]), (157,0,255), 3)
    cv2.line(roi, (src_coordenadas[2,0],src_coordenadas[2,1]), (src_coordenadas[3,0],src_coordenadas[3,1]), (157,0,255), 3)
    cv2.line(roi, (src_coordenadas[3,0],src_coordenadas[3,1]), (src_coordenadas[0,0],src_coordenadas[0,1]), (157,0,255), 3)
    #cv2.line(roi, (dst_coordenadas[0,0],dst_coordenadas[0,1]), (dst_coordenadas[1,0],dst_coordenadas[1,1]), (255,255,150), 3) 
    #cv2.line(roi, (dst_coordenadas[1,0],dst_coordenadas[1,1]), (dst_coordenadas[2,0],dst_coordenadas[2,1]), (255,255,150), 3)
    #cv2.line(roi, (dst_coordenadas[2,0],dst_coordenadas[2,1]), (dst_coordenadas[3,0],dst_coordenadas[3,1]), (255,255,150), 3)
    #cv2.line(roi, (dst_coordenadas[3,0],dst_coordenadas[3,1]), (dst_coordenadas[0,0],dst_coordenadas[0,1]), (255,255,150), 3)

    return roi,src_coordenadas,dst_coordenadas

def creacionVentanas():
    pass
    #cv2.namedWindow('Imagen real [1]',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Imagen real [1]", 200,150)

    #cv2.namedWindow('Original con area de interes',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Original con area de interes", 200,150)

    #cv2.namedWindow('Area de interes',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Area de interes", 200,150)

    #cv2.namedWindow('Eliminacion de ruido',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Eliminacion de ruido", 200,150)

    #cv2.namedWindow('Grises',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Grises", 200,150)

    #cv2.namedWindow('Blanco y negro',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Blanco y negro", 200,150)

def redimensionImagen(frame,escala):
    ancho = int(frame.shape[1] * escala / 100)
    largo = int(frame.shape[0] * escala / 100)
    nuevaDimension = (ancho,largo)

    frameResize = cv2.resize(frame, nuevaDimension, interpolation = cv2.INTER_AREA)
    return frameResize

def lineas(puntosIZQx,puntosIZQy,puntosDERx,puntosDERy,img,return_img):

    ploty = np.linspace(0, img.shape[1] - 1, img.shape[1])
    if(len(puntosDERx)>100): 
        funcionDER=np.polyfit(puntosDERx,puntosDERy,2)
        right_fitx = funcionDER[0]*ploty**2 + funcionDER[1]*ploty + funcionDER[2]
        for index in range(img.shape[0]):
            cv2.circle(return_img, (int(right_fitx[index]), int(ploty[index])), 3, (255,255,0))
    if(len(puntosIZQx)>100):
        funcionIZQ=np.polyfit(puntosIZQx,puntosIZQy,2)
        left_fitx = funcionIZQ[0]*ploty**2 + funcionIZQ[1]*ploty + funcionIZQ[2]
        for index in range(img.shape[0]):
            cv2.circle(return_img, (int(left_fitx[index]), int(ploty[index])), 3, (255,255,0))

class Nodo(object):
    def __init__(self):
        # Params
        self.Im_binaria = None
        self.blurredGauss = None
        self.br = CvBridge()            #Se crea el puente para uso de OpenCV
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)
        creacionVentanas()
        cv2.createTrackbar("Down", "Blanco y negro",  120, 255, nothing)
        # Publishers
        self.pub = rospy.Publisher('imageROI', Image,queue_size=10)

        # Subscribers
        rospy.Subscriber("/app/camera/rgb/image_raw/compressed",CompressedImage,self.callback)
        #rospy.spin()

    def callback(self, msg):
        #Callback_str = " Callback llamado en %s" % rospy.get_time()    #Mensaje en terminal
        #rospy.loginfo(Callback_str)                                    #Imprime el mensa en terminal

        #PASO 1: RECIBIR IMAGEN DE LA CAMARA                                           
        #self.cv_frame = self.br.imgmsg_to_cv2(msg, "bgr8")               #Se obtienen los frames de la camara
        np_arr = np.fromstring(msg.data, np.uint8)  
        self.cv_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 

        escala = 100 # En porcentaje
        cv_frame=redimensionImagen(self.cv_frame,escala)
        #cv2.imshow("Imagen real [1]",cv_frame)                      #Se muestra en una ventana la Imagen capturada
    
        Im_ROI,src_coordenadas,dst_coordenadas=ROI(cv_frame,escala)
        cv2.imshow("Original con area de interes",Im_ROI)
    
        Im_PerspectivaROI = perspectiva(cv_frame, src_coordenadas, dst_coordenadas)
        #cv2.imshow("Area de interes",Im_PerspectivaROI)
    
        #PASO 2: DIFUMINACION GAUSSIANA
        self.blurredGauss = difuminacionGauss(Im_PerspectivaROI, kernel=5)
        cv2.imshow("Eliminacion de ruido",self.blurredGauss)
    
        self.Im_grises=grises(self.blurredGauss)
        #cv2.imshow("Grises",Im_grises)

        Down = cv2.getTrackbarPos('Down', 'Blanco y negro')
        self.Im_binaria=BlancoNegro(self.Im_grises,Down,255)
        #cv2.imshow("Blanco y negro",self.Im_binaria)
        #cv2.imshow("Blanco y negro",self.cv_frame)
    
        #ventanas_deslizantes(Im_binaria, Im_PerspectivaROI)
        #cv2.imshow("ventanas deslizantes",Im_PerspectivaROI)
        #cv2.imshow("xxx",out_img)
        #print(leftx_base, rightx_base)
        #plt.plot(histogram)
        #plt.draw()
        #plt.pause(0.1)
        #plt.clf()
        cv2.waitKey(1)                                              #Mantiene las ventanas en pantalla



    def start(self):
        rospy.loginfo("Timing images")
        #rospy.spin()
        while not rospy.is_shutdown():
            rospy.loginfo('publishing image')
            
            if self.Im_binaria is not None:
                
                self.pub.publish(self.br.cv2_to_imgmsg(self.blurredGauss))
            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("VIDEO", anonymous=True)
    my_node = Nodo()
    my_node.start()
