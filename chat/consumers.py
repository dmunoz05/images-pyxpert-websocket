# chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os


def process_mdf_video(image):
    # Functions
    image_data = image

    def detect_bounding_box(vid):
        cwd = os.getcwd()
        directory = os.path.join(
            cwd, 'models', 'haarcascade_frontalface_default.xml')
        face_classifier = cv2.CascadeClassifier(directory)
        I_gris = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(I_gris, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        return faces

    def draw(mask, color, textColor, frame):
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

    if image is not None and 'data:image/jpeg;base64,' in image:
        image_base64 = image.replace('data:image/jpeg;base64,', '')
        decode_image = base64.b64decode(image_base64)
        # Decodificar la imagen usando OpenCV
        im_arr = np.frombuffer(decode_image, dtype=np.uint8)
        Img = cv2.imdecode(im_arr, cv2.IMREAD_UNCHANGED)

        if Img.shape[0] > 0 and Img.shape[1] > 0 and Img is not None:
            # plt.imshow(Img)
            # plt.axis("off")
            # plt.show()

            # Dibujamos los contornos
            detect_bounding_box(Img)

            buffer = BytesIO()

            # Mostrar la imagen con los contornos a color original
            imageColor = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

            # Convertir la imagen a BytesIO
            Image.fromarray(imageColor).save(buffer, format='JPEG')

            # Obtener los datos de la imagen
            image_data = buffer.getvalue()

    return image_data


def process_mdc_video(image):
    # Functions
    image_data = image

    def draw(mask, color, textColor, frame):
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

    if image is not None and 'data:image/jpeg;base64,' in image:
        image_base64 = image.replace('data:image/jpeg;base64,', '')
        decode_image = base64.b64decode(image_base64)
        # Decodificar la imagen usando OpenCV
        im_arr = np.frombuffer(decode_image, dtype=np.uint8)
        Img = cv2.imdecode(im_arr, cv2.IMREAD_UNCHANGED)
        # Img = cv2.imdecode(im_arr, cv2.IMREAD_COLOR)
        # Img = cv2.imdecode(im_arr, -1)

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
            # frameHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
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
            draw(maskRed, (0, 0, 255), 'Rojo', Img)
            draw(maskOrange, (0, 165, 255),     'Naranja', Img)
            draw(maskAmarillo, (0, 255, 255), 'Amarillo', Img)
            draw(maskVerde, (0, 255, 0), 'Verde', Img)
            draw(maskAzul, (255, 0, 0), 'Azul', Img)
            draw(maskMorado, (255, 0, 255), 'Morado', Img)
            draw(maskVioleta, (255, 0, 255), 'Violeta', Img)

            buffer = BytesIO()

            # Mostrar la imagen con los contornos a color original
            imageColor = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

            # plt.imshow(imageColor)
            # plt.axis("off")
            # plt.show()

            # Convertir la imagen a BytesIO
            Image.fromarray(imageColor).save(buffer, format='JPEG')

            # Obtener los datos de la imagen
            image_data = buffer.getvalue()

    return image_data


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = f"chat_{self.room_name}"

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
            type_model = text_data_json["type_model"]

            # Send message to room group
            await self.channel_layer.group_send(
                self.room_group_name, {"type": "chat.message",
                                       "message": message, "image_data": image_data, "type_model": type_model}
            )
        except:
            print('Error inesperado')

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]
        image_data = event["image_data"]
        type_model = event["type_model"]

        try:
            if type_model == 'mdc':
                # Model detec color
                processed_image = process_mdc_video(image_data)

            if type_model == 'mdf':
                # Model cnn
                processed_image = process_mdf_video(image_data)

            # Procesar la imagen recibida

            # Convertir a base64 de nuevo
            processed_image_base64 = base64.b64encode(
                processed_image).decode('utf-8')

            # Concatenar
            image_data = 'data:image/jpeg;base64,' + processed_image_base64

            # Enviar los datos de imagen procesados de vuelta al cliente
            await self.send(text_data=json.dumps({"message": message, "image_data": image_data}))
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
