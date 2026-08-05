
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
        # 设置六个用户
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

        
        # 设置六个用户之间的短信记录
        # 信息1和reminder配合  信息2和food配合
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
                "message": """你能帮我点一个\"玛格丽特披萨\"的外卖吗,商家是达美乐。""",
                "time": "2024-03-09",
            },
            3: {
                "sender_id": "USR102",
                "receiver_id": "USR103",
                "message": "帮我查一些喜茶有哪些奶茶外卖，买一杯便宜些的奶茶。买完以后记得回复我,回复的内容是（已经买好了）",
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

     # 发送短信
    def send_message(self, sender_name: str, receiver_name: str, message: str) -> Dict[str, Union[bool, str]]:
        """
        发送短信并添加到短信记录中。
        Args:
            sender_name (str): 发送者用户名。
            receiver_name (str): 接收者用户名。
            message (str): 要发送的短信内容。
        Returns:
            Dict[str, Union[bool, str]]: 包含发送状态和结果的字典。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法发送短信"}
        
        if self.wifi == False:
            return {"status": False, "message": "wifi关闭，此时不能发送信息"}

        if len(self.inbox) >= self.max_capacity:
            return {"status": False, "message": "内存容量不够了，你需要询问user删除哪一条短信。"}
            
        # 验证发送者和接收者是否存在
        if sender_name not in self.user_list or receiver_name not in self.user_list:
            return {"status": False, "message": "发送者或接收者不存在"}

        sender_id = self.user_list[sender_name]["user_id"]
        receiver_id = self.user_list[receiver_name]["user_id"]

        # 将短信添加到inbox
        self.message_id_counter += 1
        self.inbox[self.message_id_counter] = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message": message,
        }


        return {"status": True, "message": f"短信成功发送给{receiver_name}。"}

    # 删除短信
    def delete_message(self, message_id: int) -> Dict[str, Union[bool, str]]:
        """
        删除指定的短信。
        Args:
            message_id (int): 要删除的短信ID。
        Returns:
            Dict[str, Union[bool, str]]: 包含删除状态和结果的字典。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法删除短信"}
        if message_id not in self.inbox:
            return {"status": False, "message": "短信ID不存在"}

        del self.inbox[message_id]
        return {"status": True, "message": f"短信ID {message_id} 已成功删除。"}


    def view_messages_between_users(self, sender_name: str, receiver_name: str) -> Dict[str, Union[bool, str, List[Dict[str, str]]]]:
        """
        查看A用户发送给B用户的所有短信。
        Args:
            sender_name (str): 发送者的用户名。
            receiver_name (str): 接收者的用户名。
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, str]]]]: 包含发送者和接收者之间的短信列表。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法查看短信信息"}
        
        if sender_name not in self.user_list:
            return {"status": False, "message": "发送者不存在"}
        
        if receiver_name not in self.user_list:
            return {"status": False, "message": "接收者不存在"}

        sender_id = self.user_list[sender_name]["user_id"]
        receiver_id = self.user_list[receiver_name]["user_id"]
        messages_between_users = []

        # 遍历 inbox，找出 sender_id 发送给 receiver_id 的短信
        for msg_id, msg_data in self.inbox.items():
            if msg_data["sender_id"] == sender_id and msg_data["receiver_id"] == receiver_id:
                messages_between_users.append({
                    "id": msg_id,
                    "sender": sender_name,
                    "receiver": receiver_name,
                    "message": msg_data["message"]
                })

        if not messages_between_users:
            return {"status": False, "message": "没有找到相关的短信记录"}

        return {"status": True, "messages": messages_between_users}


    def search_messages(self, user_name: str, keyword: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        根据关键词搜索用户的短信，包括发送和接收的。
        Args:
            user_name (str): 用户名。
            keyword (str): 搜索的关键词。
        Returns:
            Dict[str, Union[str, List[Dict[str, str]]]]: 包含匹配短信的列表。
        """
        if user_name not in self.user_list:
            return {"status": False, "message": "用户不存在"}

        user_id = self.user_list[user_name]["user_id"]
        matched_messages = []

        # 遍历 inbox，找到发送或接收中包含关键词的消息
        for msg_id, msg_data in self.inbox.items():
            if (msg_data["sender_id"] == user_id or msg_data["receiver_id"] == user_id) and keyword.lower() in msg_data["message"].lower():
                matched_messages.append({
                    "id":msg_id,
                    "sender_id": msg_data["sender_id"],
                    "receiver_id": msg_data["receiver_id"],
                    "message": msg_data["message"]
                })

        if not matched_messages:
            return {"status": False, "message": "没有找到包含关键词的短信"}

        return {"status": True, "messages": matched_messages}
    
    def get_all_message_times_with_ids(self) -> Dict[int, str]:
        """
        获取所有短信的时间以及对应的短信编号。
        Returns:
            Dict[int, str]: 包含所有短信编号和时间的字典，键为短信编号，值为时间。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，获取所有短信的时间以及对应的短信编号。"}
        message_times_with_ids = {msg_id: msg_data["time"] for msg_id, msg_data in self.inbox.items()}
        return message_times_with_ids

    def get_latest_message_id(self) -> Dict[str, Union[bool, str, int]]:
        """
        获取最新发送的短信ID。
        Returns:
            Dict[str, Union[bool, str, int]]: 包含最新短信ID的字典。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法获取最新发送的短信ID。"}
        if not self.inbox:
            return {"status": False, "message": "短信记录为空"}

        # 遍历所有短信，找出时间最新的短信
        latest_message_id = None
        latest_time = None

        for message_id, message_data in self.inbox.items():
            message_time = datetime.strptime(message_data['time'], "%Y-%m-%d")
            if latest_time is None or message_time > latest_time:
                latest_time = message_time
                latest_message_id = message_id

        return {"status": True, "message": f"最新的短信ID是 {latest_message_id}", "message_id": latest_message_id}


    def get_earliest_message_id(self) -> Dict[str, Union[bool, str, int]]:
        """
        获取最早发送的短信ID。
        Returns:
            Dict[str, Union[bool, str, int]]: 包含最早短信ID的字典。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法获取最早发送的短信ID"}
        if not self.inbox:
            return {"status": False, "message": "短信记录为空"}

        # 遍历所有短信，找出时间最早的短信
        earliest_message_id = None
        earliest_time = None

        for message_id, message_data in self.inbox.items():
            message_time = datetime.strptime(message_data['time'], "%Y-%m-%d")
            if earliest_time is None or message_time < earliest_time:
                earliest_time = message_time
                earliest_message_id = message_id

        return {"status": True, "message": f"最早的短信ID是 {earliest_message_id}", "message_id": earliest_message_id}



