
from datetime import datetime
from typing import Dict, List, Union

from .base_api import BaseApi


class MessageApi(BaseApi):
    """
    A class representing a Message API for managing user interactions in a workspace.
    """

    def __init__(self):
        super().__init__()
        """
        Initialize the MessageAPI with six users and their messages.
        """
        # Initialize six users
        self.max_capacity = 6
        self.user_list: Dict[str, Dict[str, Union[str, int]]] = {
            "Eve": {
                "user_id": "USR100",
                "phone_number": "123-456-7890",
                "occupation": "Software Engineer",
            },
            "Frank": {
                "user_id": "USR101",
                "phone_number": "234-567-8901",
                "occupation": "Data Scientist",
            },
            "Grace": {
                "user_id": "USR102",
                "phone_number": "345-678-9012",
                "occupation": "Product Manager",
            },
            "Helen": {
                "user_id": "USR103",
                "phone_number": "456-789-0123",
                "occupation": "UX Designer",
            },
            "Isaac": {
                "user_id": "USR104",
                "phone_number": "567-890-1234",
                "occupation": "DevOps Engineer",
            },
            "Jack": {
                "user_id": "USR105",
                "phone_number": "678-901-2345",
                "occupation": "Marketing Specialist",
            },
        }

        
        # Initialize message history between six users
        # Message 1 pairs with reminder, Message 2 pairs with food
        self.inbox: Dict[int, Dict[str, Union[str, int]]] = {
            1: {
                "sender_id": "USR100",
                "receiver_id": "USR101",
                "message": "Hey Frank, don't forget about our meeting on 2024-06-11 at 4 PM in Conference Room 1.",
                "time": "2024-06-09",
            },
            2: {
                "sender_id": "USR101",
                "receiver_id": "USR102",
                "message": "Can you help me order a \"Margherita Pizza\" delivery? The merchant is Domino's.",
                "time": "2024-03-09",
            },
            3: {
                "sender_id": "USR102",
                "receiver_id": "USR103",
                "message": "Please check the milk tea delivery options available from Heytea and purchase a cheaper milk tea for me. After making the purchase, remember to reply to me with \"Already bought.\"",
                "time": "2023-12-05",
            },
            4: {
                "sender_id": "USR103",
                "receiver_id": "USR102",
                "message": "No problem Helen, I can assist you.",
                "time": "2024-09-09",
            },
            5: {
                "sender_id": "USR104",
                "receiver_id": "USR105",
                "message": "Isaac, are you available for a call?",
                "time": "2024-06-06",
            },
            6: {
                "sender_id": "USR105",
                "receiver_id": "USR104",
                "message": "Yes Jack, let's do it in 30 minutes.",
                "time": "2024-01-15",
            },
        }

        self.message_id_counter: int = 6
    

    def _load_scenario(self, scenario: dict, long_context=False) -> None:

        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)

    # 1. Send Message
    def send_message(self, sender_name: str, receiver_name: str, message: str) -> Dict[str, Union[bool, str]]:
        """
        Send a message and add it to the message records.
        Args:
            sender_name (str): Sender's username.
            receiver_name (str): Receiver's username.
            message (str): The content of the message to be sent.
        Returns:
            Dict[str, Union[bool, str]]: A dictionary containing the status of the send operation and the result message.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to send message"}
        
        if self.wifi == False:
            return {"status": False, "message": "Wi-Fi is turned off, cannot send messages at this time"}

        if len(self.inbox) >= self.max_capacity:
            return {"status": False, "message": "Inbox capacity is full. You need to ask the user which message to delete."}
            
        # Verify that the sender and receiver exist
        if sender_name not in self.user_list or receiver_name not in self.user_list:
            return {"status": False, "message": "Sender or receiver does not exist"}

        sender_id = self.user_list[sender_name]["user_id"]
        receiver_id = self.user_list[receiver_name]["user_id"]

        # Add the message to the inbox
        self.message_id_counter += 1
        self.inbox[self.message_id_counter] = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message": message,
        }

        return {"status": True, "message": f"Message successfully sent to {receiver_name}."}

    # 2. Delete Message
    def delete_message(self, message_id: int) -> Dict[str, Union[bool, str]]:
        """
        Delete a specified message.
        Args:
            message_id (int): The ID of the message to be deleted.
        Returns:
            Dict[str, Union[bool, str]]: A dictionary containing the status of the delete operation and the result message.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to delete message"}
        if message_id not in self.inbox:
            return {"status": False, "message": "Message ID does not exist"}

        del self.inbox[message_id]
        return {"status": True, "message": f"Message ID {message_id} has been successfully deleted."}

    # 3. View Messages Between Users
    def view_messages_between_users(self, sender_name: str, receiver_name: str) -> Dict[str, Union[bool, str, List[Dict[str, str]]]]:
        """
        View all messages sent from User A to User B.
        Args:
            sender_name (str): Sender's username.
            receiver_name (str): Receiver's username.
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, str]]]]: A dictionary containing the list of messages between the sender and receiver.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to view message information"}
        
        if sender_name not in self.user_list:
            return {"status": False, "message": "Sender does not exist"}
        
        if receiver_name not in self.user_list:
            return {"status": False, "message": "Receiver does not exist"}

        sender_id = self.user_list[sender_name]["user_id"]
        receiver_id = self.user_list[receiver_name]["user_id"]
        messages_between_users = []

        # Iterate through the inbox to find messages sent from sender_id to receiver_id
        for msg_id, msg_data in self.inbox.items():
            if msg_data["sender_id"] == sender_id and msg_data["receiver_id"] == receiver_id:
                messages_between_users.append({
                    "id": msg_id,
                    "sender": sender_name,
                    "receiver": receiver_name,
                    "message": msg_data["message"]
                })

        if not messages_between_users:
            return {"status": False, "message": "No related message records found"}

        return {"status": True, "messages": messages_between_users}


    def search_messages(self, user_name: str, keyword: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Search a user's messages based on a keyword, including sent and received messages.
        Args:
            user_name (str): Username.
            keyword (str): Keyword to search.
        Returns:
            Dict[str, Union[str, List[Dict[str, str]]]]: List of matching messages.
        """
        if user_name not in self.user_list:
            return {"status": False, "message": "User does not exist"}

        user_id = self.user_list[user_name]["user_id"]
        matched_messages = []

        # Iterate through the inbox to find messages sent or received by the user that contain the keyword
        for msg_id, msg_data in self.inbox.items():
            if (msg_data["sender_id"] == user_id or msg_data["receiver_id"] == user_id) and keyword.lower() in msg_data["message"].lower():
                matched_messages.append({
                    "id": msg_id,
                    "sender_id": msg_data["sender_id"],
                    "receiver_id": msg_data["receiver_id"],
                    "message": msg_data["message"]
                })

        if not matched_messages:
            return {"status": False, "message": "No messages found containing the keyword"}

        return {"status": True, "messages": matched_messages}

    def get_all_message_times_with_ids(self) -> Union[Dict[int, str], Dict[str, Union[bool, str]]]:
        """
        Get the times of all messages along with their corresponding message IDs.
        Returns:
            Union[Dict[int, str], Dict[str, Union[bool, str]]]: A dictionary with message IDs as keys and times as values, or an error message.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to retrieve all message times and their corresponding message IDs."}
        message_times_with_ids = {msg_id: msg_data["time"] for msg_id, msg_data in self.inbox.items()}
        return message_times_with_ids

    def get_latest_message_id(self) -> Dict[str, Union[bool, str, int]]:
        """
        Get the ID of the latest sent message.
        Returns:
            Dict[str, Union[bool, str, int]]: A dictionary containing the latest message ID.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to retrieve the latest sent message ID."}
        if not self.inbox:
            return {"status": False, "message": "No message records found"}

        # Iterate through all messages to find the latest message based on time
        latest_message_id = None
        latest_time = None

        for message_id, message_data in self.inbox.items():
            message_time = datetime.strptime(message_data['time'], "%Y-%m-%d")
            if latest_time is None or message_time > latest_time:
                latest_time = message_time
                latest_message_id = message_id

        return {"status": True, "message": f"The latest message ID is {latest_message_id}", "message_id": latest_message_id}

    def get_earliest_message_id(self) -> Dict[str, Union[bool, str, int]]:
        """
        Get the ID of the earliest sent message.
        Returns:
            Dict[str, Union[bool, str, int]]: A dictionary containing the earliest message ID.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in, unable to retrieve the earliest sent message ID."}
        if not self.inbox:
            return {"status": False, "message": "No message records found"}

        # Iterate through all messages to find the earliest message based on time
        earliest_message_id = None
        earliest_time = None

        for message_id, message_data in self.inbox.items():
            message_time = datetime.strptime(message_data['time'], "%Y-%m-%d")
            if earliest_time is None or message_time < earliest_time:
                earliest_time = message_time
                earliest_message_id = message_id

        return {"status": True, "message": f"The earliest message ID is {earliest_message_id}", "message_id": earliest_message_id}



