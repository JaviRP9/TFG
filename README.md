# Trabajo de Fin de Grado

Este repositorio contiene el codigo correspondiente al Trabajo Fin de Grado "Mejora de la precisión de un sistema de localización en tiempo real". 

## Contenido

### Servidor UDP

Servidor implementado para la recepción de paquetes UDP enviados por los sensores UWB DMW1001. 

### Filtro de Kalman

Implementación en Python de una clase con los metodos correspondientes para el uso del filtro de Kalman.

### Realtime System 

Codigo implementado del sistema propuesto. Este esta formado por la recepción de las coordenadas de los sensores a través de MQTT, la recepción de aceleraciones a través de UDP y el post procesado necesario para la implementación del filtro de kalman. 