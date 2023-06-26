import socket
import struct

local_ip="172.17.0.1"
local_port=4242
bufferSize=2048

#Create a datagram socket
sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

#Bind
server = (local_ip, local_port)
sock.bind(server)

print("Listening on " + local_ip + ":" + str(local_port))

#Listen

while (True):
    payload, client_address = sock.recvfrom(bufferSize)
    len_payload=len(payload)
    print("Message received (" + str(len_payload) + " bytes) from: " + str(client_address))
    data=[payload[i:i + 4] for i in range(0,len_payload,4)]
    byte0_val1=data[0]
    byte0_val2=data[1]
    byte1_val1=data[2]
    byte1_val2=data[3]
    byte2_val1=data[4]
    byte2_val2=data[5]
    byte3_euid1=data[6]
    byte3_euid2=data[7]
    euid1=hex(struct.unpack("i", byte3_euid1)[0])
    euid2=hex(struct.unpack("i", byte3_euid2)[0])
    euid=euid1 + euid2.replace("0x","")
    x=float(int(struct.unpack("i", byte0_val1)[0])) + float(int(struct.unpack("i", byte0_val2)[0])) / 1000000
    y=float(int(struct.unpack("i", byte1_val1)[0])) + float(int(struct.unpack("i", byte1_val2)[0])) / 1000000
    z=float(int(struct.unpack("i", byte2_val1)[0])) + float(int(struct.unpack("i", byte2_val2)[0])) / 1000000
    print("EUID " + euid + " Accelerometer x: " + format(x) + " y: " + format(y) + " z: " + format(z))