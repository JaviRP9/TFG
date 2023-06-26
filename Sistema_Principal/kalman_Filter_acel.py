import math
import numpy as np

class kalmanFilter_acel(object):
        '''
        ######## Parametros ###########

        dim_x : int
            -> Numero de variables de estado del filtro de kalman

        dim_ z : int
            -> Numero de entrada al sistema. Por ejemplo:
            coordenas (x,y) seria dim_z = 2
            coordenadas (x,y,z) seria dim_z = 3

        coeff : float
            -> Coeficiente de ruido de medida

        ###############################

        ######## Atributos ############

        X : np.array(dim_x, 1)
            -> Matriz de estado

        P : np.eye(dim_x)
            -> Matriz de covarianza del proceso
               Inicialmente diagonal, ya que se dice que no existe relación
               entre ninguna de las variables de estado 

        A : np.eye(dim_x)
            -> Matriz de trasición de estados

        H : np.eye(dim_x)
            -> Matriz de observación 

        KG : np.array(dim_x, dim_z)
            -> Ganancia de Kalman

        I : np.eye(dim_x)
            -> Matriz Identidad

        Q : np.eye(dim_x)
            -> Matriz de covarianza de ruido del proceso

        R : np.eye(dim_x)
            -> Matriz de covarianza de la medida

        Sum_Q : np.zeros(dim_x)
            -> Matriz auxiliar para el calculo del la Matriz de covarianza de ruido del proceso

        Sum_R : np.zeros(dim_x)
            -> Matriz auxiliar para el calculo del la Matriz de covarianza de la medida

        Sum_mean_Q : np.zeros((dim_x,1))
            -> Matriz auxiliar para el calculo del la Matriz de covarianza de ruido del proceso


        Sum_mean_R : np.zeros((dim_x,1))
            -> Matriz auxiliar para el calculo del la Matriz de covarianza de la medida

        Z : np.ones((dim_x,1))
            -> Ruido de la medida

        ###############################

        ## X_k_ant : Matriz de estato de la iteracion anterior
        '''
        ###############################
        def __init__(self, dim_x, dim_z, coeff):
            
            #Definimos las distintas variables de la clase
            self.dim_x = dim_x
            self.dim_z = dim_z
            self.X = np.zeros((dim_x,1)) # [x, y, vx, vy, ax, ay] # State Matrix  # x = Fx + Bu
            self.P = np.eye(dim_x)   # Procces Covariance Matrix
            #self.A = np.eye(dim_x)
            dt = 0.1
            self.A = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                              [0, 1, 0, dt, 0, 0.5*dt**2],
                              [0, 0, 1, 0, dt, 0],
                              [0, 0, 0, 1, 0, dt],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
            self.H = np.eye(dim_x)
            self.KG = np.zeros((dim_x, dim_z)) # kalman gain
            #self.y =  np.zeros((dim_z, 1))
            #self.S = np.zeros((dim_z, dim_z)) # system uncertainty
            #self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty
            self.I = np.eye(dim_x)
            self.Q = np.eye(dim_x)             # Noise Covariance Matrix
            self.R = np.eye(dim_x)
            self.sum_Q = np.zeros(dim_x)
            self.sum_R = np.zeros(dim_x)
            self.sum_mean_Q = np.zeros((dim_x,1))
            self.sum_mean_R = np.zeros((dim_z,1))
            #Z = 0.01 * np.ones((dim_x,1)) ----> z ES EL RUIDO DE LA MEDIDA
            self.Z = coeff * np.ones((dim_x,1))


        def predict(self):

            '''
            Predicción del nuevo estado

            '''

            self.X = np.dot(self.A,self.X)    #State Matrix
            self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q #Process Covariance Matrix

        def update(self, measure):

            '''
            Actualización con la nueva medida y la ganancia de Kalman

            ######## Parametros ###########

            measure : np.array((x, y))
                -> Vector de coordenadas del sistema

            ###############################

            ######## Aclaraciones #########

            y : medida del nuevo estado
                -> Calculo como en la documentación teorica , Y_k = H*X_k + Z
                Siendo Z el ruido de medida

            I_KH : I - KG*H
                -> Explicado en la documentación teorica

            Y_error : Y - H*X_k
                -> Error entre la medida y la prediccón. Explicado en la documentación teorica

            ###############################
            '''

            y = np.dot(self.H, measure) + self.Z
            y_err = y - np.dot(self.H, self.X)

            #Kalman Gain
            S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
            self.KG = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            I_KH = self.I - np.dot(self.KG, self.H)
            X_k = self.X + np.dot(self.KG, y_err) # Ecuaciones: # X_k = X_Kp + KG*Y
            P_k = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.KG, self.R),self.KG.T)

            #Guardamos estado anterior para calculo de Q y R
            X_p_ant = self.X
            measure_ant = measure
            #Guardamos medida y posterior estado
            self.X = X_k
            self.P = P_k

            return X_p_ant, measure_ant

        def R_covarianza_matrix(self, measure, measure_ant, n):

            '''
            Función para estimar la matriz de covarianza de la medida. 

            ######## Parametros ###########

            measure : np.array((x, y))
                -> Vector de coordenadas del sistema

            measure_ant : np.array((x-1, y-1))
                -> Vector de coordenadas del sistema de la iteración anterior

            n : int
                -> Numero de iteración

            ###############################
            '''
            self.sum_mean_R = self.sum_mean_R + (measure - measure_ant)
            self.mean_R = self.sum_mean_R / n
            self.sum_R = self.sum_R + np.dot((measure - measure_ant - self.mean_R),(measure - measure_ant - self.mean_R).T)
            self.R = self.sum_R / (n-1)


        def Q_covarianza_matrix(self, X_k_ant, n):

            '''
            Función para estimar la matriz de  de covarianza de ruido del proceso

            ######## Parametros ###########

            X_k_ant : self.X
                -> Matriz de estado del proceso de la iteración anterior

            n : int
                -> Numero de iteración

            ###############################
            '''

            self.sum_mean_Q = self.sum_mean_Q + (self.X - X_k_ant)
            self.mean_Q = self.sum_mean_Q / n
            self.sum_Q = self.sum_Q + np.dot((self.X - X_k_ant - self.mean_Q),(self.X - X_k_ant - self.mean_Q).T)
            self.Q = self.sum_Q / (n-1)

        def rmse(x,y,x_k,y_k):

            '''
            Función para estimar el error entre la señal antes del filtro y despues del filtro

            ######## Parametros ###########

            x : np.array(x)
                -> Vector de coordenadas x del sistema

            y : np.array(y)
                -> Vector de coordenadas y del sistema

            x_k : np.array(x_k)
                -> Vector de coordenas x del pues del filtro de Kalman (Solucion)

            y_k : np.array(y_k)
                -> Vector de coordenas y del pues del filtro de Kalman (Solucion)

            ###############################
            '''

            sum_x = 0
            sum_y = 0
            for i in range(len(x)):
                sum_x = sum_x + (x[i] - x_k[i])**2
                sum_y = sum_y + (y[i] - y_k[i])**2
            rmse_x = np.sqrt((1/len(x)) * sum_x)
            rmse_y = np.sqrt((1/len(y)) * sum_y)

            return rmse_x + rmse_y
