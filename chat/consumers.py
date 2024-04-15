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
        try:
            text_data_json = json.loads(text_data)
            message = text_data_json["message"]
            image_data = text_data_json["image_data"]

            # Send message to room group
            await self.channel_layer.group_send(
                self.room_group_name, {"type": "chat.message", "message": message, "image_data": image_data}
            )
        except:
            print('Error inesperado')

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]
        image_data = event["image_data"]

        # Send message to WebSocket
        await self.send(text_data=json.dumps({"message": message, "image_data": image_data}))