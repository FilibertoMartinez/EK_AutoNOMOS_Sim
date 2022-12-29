#!/usr/bin/env python3

import tensorflow as tf
import rospy
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

from std_msgs.msg import Int16

def redimensionImagen(frame,escala):
    ancho = int(frame.shape[1] * escala / 100)
    largo = int(frame.shape[0] * escala / 100)
    nuevaDimension = (ancho,largo)

    frameResize = cv2.resize(frame, nuevaDimension, interpolation = cv2.INTER_AREA)
    return frameResize

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
    #cv2.line(roi, (src_coordenadas[0,0],src_coordenadas[0,1]), (src_coordenadas[1,0],src_coordenadas[1,1]), (157,0,255), 3) 
    #cv2.line(roi, (src_coordenadas[1,0],src_coordenadas[1,1]), (src_coordenadas[2,0],src_coordenadas[2,1]), (157,0,255), 3)
    #cv2.line(roi, (src_coordenadas[2,0],src_coordenadas[2,1]), (src_coordenadas[3,0],src_coordenadas[3,1]), (157,0,255), 3)
    #cv2.line(roi, (src_coordenadas[3,0],src_coordenadas[3,1]), (src_coordenadas[0,0],src_coordenadas[0,1]), (157,0,255), 3)

    return roi,src_coordenadas,dst_coordenadas

def perspectiva(img, src_coordenadas=None, dst_coordinates=None):
    
    img_size = (img.shape[1], img.shape[0])

    # Se obtiene la transformacion de perspectiva con las coordenadas de ROI y destino
    M = cv2.getPerspectiveTransform(src_coordenadas, dst_coordinates)

    # Se obtiene la transformacion de perspectiva inversa
    Minv = cv2.getPerspectiveTransform(dst_coordinates, src_coordenadas)
    
    # Se crea la perspectiva con interpolacion lineal
    cambioPerspectiva = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return cambioPerspectiva

def difuminacionGauss(image, kernel=10):
    '''
    Funcion para reducir el ruido en imagenes mediante una funcion Gaussiana [En espacio X,Y]
    '''
    difuminacion = cv2.GaussianBlur(image, (kernel,kernel), 0)
    return difuminacion

def grises(image):
    grises = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grises

def callback(msg):
    global Im_grises

    #np_arr = np.fromstring(msg.data, np.uint8)
    np_arr = np.frombuffer(msg.data, np.uint8)  
    cv_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
    #cv2.imshow("vista",cv_frame)
    escala = 50 # En porcentaje
    cv_frame=redimensionImagen(cv_frame,escala)
    #cv2.imshow("Imagen real [1]",cv_frame)                      #Se muestra en una ventana la Imagen capturada
    
    Im_ROI,src_coordenadas,dst_coordenadas=ROI(cv_frame,escala)
    #cv2.imshow("Original con area de interes",Im_ROI)
    
    Im_PerspectivaROI = perspectiva(cv_frame, src_coordenadas, dst_coordenadas)
    #cv2.imshow("Area de interes",Im_PerspectivaROI)
    
    #PASO 2: DIFUMINACION GAUSSIANA
    blurredGauss = difuminacionGauss(Im_PerspectivaROI, kernel=5)
    #cv2.imshow("Eliminacion de ruido",blurredGauss)
    
    #Im_grises=grises(blurredGauss)
    Im_grises=blurredGauss
    cv2.imshow("Imagen a procesar",Im_grises)
    cv2.waitKey(1)                                              #Mantiene las ventanas en pantalla

if __name__ == "__main__":
    br = CvBridge()            #Se crea el puente para uso de OpenCV
    
    scan_sub = rospy.Subscriber('/app/camera/rgb/image_raw/compressed', CompressedImage, callback)
    vel_pub=rospy.Publisher('/AutoNOMOS_mini/manual_control/speed',Int16,queue_size=10)
    angle_pub=rospy.Publisher('/AutoNOMOS_mini/manual_control/steering',Int16,queue_size=10)

    rospy.init_node('MOVE')

    state_change_time = rospy.Time.now()
    rate = rospy.Rate(10)

    model = tf.keras.models.load_model('/home/filiberto/Documentos/modelSimu.h5')
    

    while not rospy.is_shutdown():
        vel_pub.publish(-500)
        Im_grises=Im_grises/255.
        print(Im_grises.shape)
        Im_grises=cv2.resize(Im_grises, (64,64), interpolation = cv2.INTER_AREA)
        Impredict=np.array(Im_grises).reshape((1,64,64,3)) 
        predictIm = model.predict(Impredict)
        predictIm2=predictIm*180
        print(np.sum(predictIm2).astype(int))
        #print(predictIm2[0].astype(float))
        #angle_pub.publish(predictIm2[0].astype(int))
        angle_pub.publish(np.sum(predictIm2).astype(int))
        rate.sleep()
