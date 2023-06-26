import paho.mqtt.client as mqtt
import json
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from kalman_Filter_acel import kalmanFilter_acel 
import threading
import socket

local_ip="192.168.1.36"
local_port=4242
bufferSize=2048
euid_bytes = 8
session_rand_bytes = 4
ctr_bytes = 4

topic = "location/coordinates/#"
host = "192.168.1.40"
port = 1883
node_data_anchor = {}
node_data_tag = {}
# Tags prameters
x_values_tag = []
y_values_tag = []
x_vel_tag = []
y_vel_tag = []
x_acel_tag = []
y_acel_tag = []
timestamp_tag = []
tags = [ "0x25A1DB2929C1" ] # En data esta en el 0x, si no se incluye no detectará los tags
stats_dic = {}
dt = 100 # ms

# Variables del filtro de kalman
x_kalman = []
y_kalman = []
z_kalman = []
dim_x = 6 # [x, y, vx, vy, ax, ay]
dim_z = 6
coeff_measure = 0   #Coeficiente de error en la medida
n_iteraciones = 0   #Iteraciones del filtro de kalman

## UDP SEVER ##
# Create a datagram socket
sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

#Bind
server = (local_ip, local_port)
sock.bind(server)
###############

def udp_acell():

    while(True):
        payload, client_address = sock.recvfrom(bufferSize)
        len_payload=len(payload)
        euid = hex(int.from_bytes(payload[0:euid_bytes], byteorder='little', signed=False)).replace("0x","").upper()
        session_rand = int.from_bytes(payload[euid_bytes: euid_bytes + session_rand_bytes], byteorder='little', signed=False)
        ctr = int.from_bytes(payload[euid_bytes + session_rand_bytes: euid_bytes + session_rand_bytes + ctr_bytes], byteorder='little', signed=False)
        data = [payload[i:i + 4] for i in range(euid_bytes + session_rand_bytes + ctr_bytes,len_payload,4)]
        x_val1 = data[0]
        x_val2 = data[1]
        y_val1 = data[2]
        y_val2 = data[3]
        z_val1 = data[4]
        z_val2 = data[5]
        x = float(int.from_bytes(x_val1, byteorder='little', signed=True)) + float(int.from_bytes(x_val2, byteorder='little', signed=True)) / 1000000
        y = float(int.from_bytes(y_val1, byteorder='little', signed=True)) + float(int.from_bytes(y_val2, byteorder='little', signed=True)) / 1000000
        z = float(int.from_bytes(z_val1, byteorder='little', signed=True)) + float(int.from_bytes(z_val2, byteorder='little', signed=True)) / 1000000

        if euid in stats_dic:
            if stats_dic[euid][0] != session_rand:
                stats_dic[euid] = [session_rand, ctr, 1, 0, 0, len(payload), 0, time.time(), x, y, z]
            else:
                stats_dic[euid][3] += ctr - stats_dic[euid][1] - 1
                stats_dic[euid][1] = ctr
                stats_dic[euid][2] += 1
                stats_dic[euid][4] = 100*stats_dic[euid][3]/(stats_dic[euid][3] + stats_dic[euid][2])
                stats_dic[euid][5] += len(payload)
                stats_dic[euid][6] = stats_dic[euid][5]/(time.time() - stats_dic[euid][7])
                stats_dic[euid][8] = x
                stats_dic[euid][9] = y
                stats_dic[euid][10] = z
        else:
            stats_dic[euid] = [session_rand, ctr, 1, 0, 0, len(payload), 0, time.time(), x, y, z]

        print("UDP ACELEROMETER, EUID: ", euid, " X: ", stats_dic[euid][8], " Y: ", stats_dic[euid][9], " Z: ", stats_dic[euid][10])

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(topic)

def on_message(client, userdata, msg):
    # Formato de datos recibidos : {"uid": "825A8973B99B", "x": 0, "y": 0, "z": 0, "timestamp": 21432484}
    data = json.loads(msg.payload.decode("utf-8"))
    uid = data['uid']

    if uid in (node_data_tag or node_data_anchor):
        if uid in tags:
            node_data_tag[uid].clear()
            node_data_tag[uid].append(json.loads(msg.payload.decode("utf-8")))
        else:
            node_data_anchor[uid].clear()
            node_data_anchor[uid].append(json.loads(msg.payload.decode("utf-8")))
    else:
        if uid in tags:
            node_data_tag[uid] = [json.loads(msg.payload.decode("utf-8"))]
        else:
            node_data_anchor[uid] = [json.loads(msg.payload.decode("utf-8"))]

def update_filter(KF, coord_tag, coor_tag_ant):

    '''
    Coord_tag tiene que ser un array con las coordenadas
    '''
    global n_iteraciones
    n_iteraciones = n_iteraciones + 1

    # Predict
    kalmanFilter_acel.predict(KF)
    # Update
    X_k_ant = kalmanFilter_acel.update(KF, coord_tag)

    if(n_iteraciones >= 2):
        kalmanFilter_acel.R_covarianza_matrix(KF, coord_tag, coor_tag_ant, n_iteraciones)
        kalmanFilter_acel.Q_covarianza_matrix(KF, X_k_ant, n_iteraciones)

def update(frame):
    global stats_dic
    ax.clear()
    x_values_anchor = []
    y_values_anchor = []
    data_list_anchor = list(node_data_anchor.values())
    data_list_tag = list(node_data_tag.values())
    global n_iteraciones

    # Anchors Plotting
    for i, data in enumerate(data_list_anchor):
        x_values_anchor.append(data[0]['x'])
        y_values_anchor.append(data[0]['y'])
        ax.annotate(str(i+1),xy=(data[0]['x'],data[0]['y']),xytext=(5,5),textcoords='offset points') # Con textcoords='offset points' especificamos el xytext y e ajustara a 5 puntos arriba y 5 a la derecha
        
    # Actualizar la posición de los puntos en la gráfica
    ax.scatter(x_values_anchor,y_values_anchor,color='red')

    # Actualizar la posición de los puntos en la gráfica
    
    # Tag Plotting
    for i, data in enumerate(data_list_tag):
        x_values_tag.append(data[0]['x'])
        y_values_tag.append(data[0]['y'])
        uid = (data[0]['uid']).replace('0x','')

        # Guardamos coordenadas del tag de la iteracion actual, ya que hace falta para el filtro de kalman
        tag_coord = np.array([[x_values_tag[-1]], [y_values_tag[-1]], [0], [0], [0], [0]])
        if len(x_values_tag) > 1:
            x_vel_tag.append((x_values_tag[-2] - x_values_tag[-1]) / dt)
            y_vel_tag.append((y_values_tag[-2] - y_values_tag[-1]) / dt)
            x_acel_tag.append(stats_dic[uid][8])
            y_acel_tag.append(stats_dic[uid][9])
            tag_coord = np.array([[x_values_tag[-1]], [y_values_tag[-1]], [x_vel_tag[-1]], [y_vel_tag[-1]], [x_acel_tag[-1]], [y_acel_tag[-1]]])
            if len(x_vel_tag) > 1:
                tag_coord_ant = np.array([[x_values_tag[-2]], [y_values_tag[-2]], [x_vel_tag[-2]], [y_vel_tag[-2]], [x_acel_tag[-2]], [y_acel_tag[-2]]])
            else:
                tag_coord_ant = np.array([[x_values_tag[-2]], [y_values_tag[-2]], [0], [0], [0], [0]])

            update_filter(Filter,tag_coord, tag_coord_ant)

        # Para no contar el primer valor del filtro, ya que da un resultado no correcto y optimo
        if(n_iteraciones > 2):
            x_kalman.append(Filter.X[0])
            y_kalman.append(Filter.X[1])
        if len(x_values_tag) > 1:
            ax.plot(x_values_tag, y_values_tag, color='b', label='Trayectoria UWB')
            ax.plot(x_kalman, y_kalman, color='g', label='Trayectoria con KF')
            ax.legend()

    # Establecer los límites de los ejes
    ax.set_xlim(min(x_values_anchor, default= 0)-20, max(x_values_anchor, default= 0)+20)
    ax.set_ylim(min(y_values_anchor, default= 0)-20, max(y_values_anchor, default= 0)+20)

    # Establecer el título y las etiquetas de los ejes
    ax.set_title('Posición actual de los nodos')
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')

# Iniciamos MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(host, port, 60)
client.loop_start() # Con loop_start el programa puede continuar ejecutandose

# Filtro de Kalman
Filter = kalmanFilter_acel(dim_x, dim_z, coeff_measure)

# Thread
thread_udp = threading.Thread(target=udp_acell)
thread_udp.start()

# Grafica
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, interval=100)
plt.show()
