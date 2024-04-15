# chat/consumers.py
import json
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer
from asgiref.sync import async_to_sync
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def process_video(image):
    # Functions
    image_data = image

    def dibujar(mask, color, textColor, frame):
        contornos, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 2000:
                M = cv2.moments(c)
                if (M["m00"] == 0):
                    M["m00"] = 1
                x = int(M["m10"]/M["m00"])
                y = int(M["m01"]/M["m00"])
                cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, textColor, (x+10, y), font,
                            0.75, (0, 255, 0), 1, cv2.LINE_AA)

                nuevoContorno = cv2.convexHull(c)
                cv2.drawContours(frame, [nuevoContorno], 0, color, 3)

    if image is not None and 'data:image/jpeg;base64' in image:
        image_base64 = image.replace('data:image/jpeg;base64,', '')
        decode_image = base64.b64decode(image_base64)
        # Decodificar la imagen usando OpenCV
        im_arr = np.frombuffer(decode_image, dtype=np.uint8)
        Img = cv2.imdecode(im_arr, -1)

        # Colores HSV
        # Rojo
        redBajo1 = np.array([0, 100, 20], np.uint8)
        redAlto1 = np.array([5, 255, 255], np.uint8)
        redBajo2 = np.array([175, 100, 20], np.uint8)
        redAlto2 = np.array([180, 255, 255], np.uint8)

        # Naranja
        orangeBajo = np.array([5, 100, 20], np.uint8)
        orangeAlto = np.array([15, 255, 255], np.uint8)

        # Amarillo
        amarilloBajo = np.array([15, 100, 20], np.uint8)
        amarilloAlto = np.array([45, 255, 255], np.uint8)

        # Verde
        verdeBajo = np.array([45, 100, 20], np.uint8)
        verdeAlto = np.array([85, 255, 255], np.uint8)

        # Azul claro
        azulBajo1 = np.array([100, 100, 20], np.uint8)
        azulAlto1 = np.array([125, 255, 255], np.uint8)

        # Azul oscuro
        azulBajo2 = np.array([125, 100, 20], np.uint8)
        azulAlto2 = np.array([130, 255, 255], np.uint8)

        # Morado
        moradoBajo = np.array([135, 100, 20], np.uint8)
        moradoAlto = np.array([145, 255, 255], np.uint8)

        # Violeta
        violetaBajo = np.array([145, 100, 20], np.uint8)
        violetaAlto = np.array([170, 255, 255], np.uint8)

        if Img.shape[0] > 0 and Img.shape[1] > 0 and Img is not None:
            frameHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
            # Detectamos los colores

            # Rojo
            maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
            maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
            maskRed = cv2.add(maskRed1, maskRed2)

            # Naranja
            maskOrange = cv2.inRange(frameHSV, orangeBajo, orangeAlto)

            # Amarillo
            maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)

            # Verde
            maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)

            # Azul
            maskAzul1 = cv2.inRange(frameHSV, azulBajo1, azulAlto1)
            maskAzul2 = cv2.inRange(frameHSV, azulBajo2, azulAlto2)
            maskAzul = cv2.add(maskAzul1, maskAzul2)

            # Morado
            maskMorado = cv2.inRange(frameHSV, moradoBajo, moradoAlto)

            # Violeta
            maskVioleta = cv2.inRange(frameHSV, violetaBajo, violetaAlto)

            # Dibujamos los contornos
            dibujar(maskRed, (0, 0, 255), 'Rojo', Img)
            dibujar(maskOrange, (0, 165, 255), 'Naranja', Img)
            dibujar(maskAmarillo, (0, 255, 255), 'Amarillo', Img)
            dibujar(maskVerde, (0, 255, 0), 'Verde', Img)
            dibujar(maskAzul, (255, 0, 0), 'Azul', Img)
            dibujar(maskMorado, (255, 0, 255), 'Morado', Img)
            dibujar(maskVioleta, (255, 0, 255), 'Violeta', Img)

            buffer = BytesIO()

            # Convertir la imagen a BytesIO
            Image.fromarray(Img).save(buffer, format='PNG')

            # Obtener los datos de la imagen
            image_data = buffer.getvalue()

    return image_data


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = f"chat_{self.room_name}"

        self.id = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.id
        self.user = self.scope['user']

        # print("Id : ", self.id)
        # print("Conexión establecida room_group_name: ", self.room_group_name)
        # print("Conexión establecida channel_name: ", self.channel_name)

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # Receive message from WebSocket
    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            message = text_data_json["message"]
            image_data = text_data_json["image_data"]

            # Dibujar
            # image_process = await ProcessVideo.run(image_data)

            # Send message to room group
            await self.channel_layer.group_send(
                self.room_group_name, {"type": "chat.message",
                                       "message": message, "image_data": image_data}
            )
        except:
            print('Error inesperado')

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]
        image_data = event["image_data"]

        try:
            # Procesar la imagen recibida
            processed_image = process_video(image_data)

            # Convertir a base64 de nuevo
            processed_image_base64 = base64.b64encode(processed_image).decode('utf-8')

            # Concatenar
            image_data = 'data:image/jpeg;base64,' + processed_image_base64

            # Enviar los datos de imagen procesados de vuelta al cliente
            await self.send(text_data=json.dumps({"message": message, "image_data": image_data}))
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")