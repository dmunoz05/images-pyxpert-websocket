# chat/consumers.py
from datetime import timezone
import json
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer
from asgiref.sync import async_to_sync


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = f"chat_{self.room_name}"

        self.id = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.id
        self.user = self.scope['user']

        print("Id : ", self.id)
        print("Conexión establecida room_group_name: ", self.room_group_name)
        print("Conexión establecida channel_name: ", self.channel_name)

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name, {"type": "chat.message", "message": message}
        )

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]

        # Send message to WebSocket
        await self.send(text_data=json.dumps({"message": message}))


# class ChatConsumer(WebsocketConsumer):
#     def connect(self):
#         self.id = self.scope['url_route']['kwargs']['room_name']
#         self.room_group_name = 'chat_%s' % self.id
#         self.user = self.scope['user']
#         print("Id : ", self.id)
#         print("Conexión establecida room_group_name: ", self.room_group_name)
#         print("Conexión establecida channel_name: ", self.channel_name)

#         async_to_sync(self.channel_layer.group_add)(
#             self.room_group_name, self.channel_name)
#         self.accept()

#     def disconnect(self, close_code):
#         print('Se ha desconectado')
#         async_to_sync(self.channel_layer.group_discard)(
#             self.room_group_name, self.channel_name)
#         pass

#     def receive(self, text_data):
#         try:
#             print('Mensaje recibido')
#             text_data_json = json.loads(text_data)
#             message = text_data_json["message"]

#             # Obtener el Id del usuario que envía el mensaje
#             if self.scope['user']:
#                 sender_id = self.scope['user'].id
#                 print('Sender_id:', sender_id)
#                 async_to_sync(self.channel_layer.group_send)(self.room_group_name, {
#                     'type': 'chat_message',
#                     'message': message,
#                     'username': self.scope['user'].username,
#                     'datetime': timezone.localtime(timezone.now()).strftime('%Y-%m-%d %H:%M:%S'),
#                     'sender_id': sender_id,
#                 })
#             else:
#                 print('El usuario no está autenticado. Ignorando el mensaje')

#         except json.JSONDecodeError as e:
#             print('Error en la codificación del JSON: ', e)
#         except KeyError as e:
#             print('Error en la clave del JSON: ', e)
#         except Exception as e:
#             print('Error desconocido: ', e)

#     def chat_message(self, event):
#         message = event['message']
#         username = event['username']
#         datetime = event['datetime']
#         sender_id = event['sender_id']

#         current_user_id = self.scope['user'].id

#         if sender_id != current_user_id:
#             self.send(text_data=json.dumps({
#                 'message': message,
#                 'username': username,
#                 'datetime': datetime
#             }))
