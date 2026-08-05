
from datetime import datetime
from typing import Dict, List, Union

from .base_api import BaseApi


class ReminderApi(BaseApi):
    """
    A class representing a Reminder API for managing reminders and notifications in a system.
    """

    def __init__(self):
        """
        Initialize the ReminderAPI with some predefined reminders.
        """
        self.max_capacity = 6  
        self.reminder_list: Dict[int, Dict[str, Union[str, bool, datetime]]] = {
            1: {
                "reminder_id": 1001,
                "title": "Doctor's Appointment",
                "description": "Visit Dr. Smith for a checkup.",
                "time": "2024-07-15 09:30",
                "notified": False,
            },
            2: {
                "reminder_id": 1002,
                "title": "Team Meeting",
                "description": "Monthly project review with the team.",
                "time": "2024-07-17 11:00",
                "notified": False,
            },
            3: {
                "reminder_id": 1003,
                "title": "To-do list",
                "description": """首先帮Frank在"盒马生鲜"点外卖，需要定两个"生鲜大礼包"，再发短信告诉Frank："购买商品的价格是()元"。要把括号换成实际金额，保留一位小数。""",
                "time": "2024-07-16 11:00",
                "notified": False,
            },
        }
        self.reminder_id_counter: int = 3
    

    def _load_scenario(self, scenario: dict, long_context=False) -> None:

        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)

    def _check_capacity(self) -> bool:
        """
        检查备忘录容量是否已满。
        Returns:
            bool: 如果容量已满，返回 True；否则返回 False。
        """
        return len(self.reminder_list) >= self.max_capacity
    
    def view_reminder_by_title(self, title: str) -> Dict[str, Union[str, bool, Dict[str, Union[str, bool, datetime]]]]:
        """
        根据提醒的标题查看特定的提醒。
        Args:
            title (str): 提醒的标题。
        Returns:
            Dict[str, Union[str, bool, Dict]]: 包含查找状态和提醒详情的字典。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法查看提醒"}
        for reminder_id, reminder in self.reminder_list.items():
            if reminder["title"] == title:
                return {"status": True, "reminder": reminder}
        
        return {"status": False, "message": f"没有找到标题为 '{title}' 的提醒"}


    # 添加提醒
    def add_reminder(self, title: str, description: str, time: datetime) -> Dict[str, Union[bool, str]]:
        """
        添加一个新的提醒。
        Args:
            title (str): 提醒标题。
            description (str): 提醒描述。
            time (datetime): 提醒时间。
        Returns:
            Dict[str, Union[bool, str]]: 包含添加状态和结果的字典。
        """

        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法添加一个新的提醒"}
        if self._check_capacity():
            return {"status": False, "message": "提醒容量已满，无法添加新的提醒"}

        self.reminder_id_counter += 1
        reminder_id = self.reminder_id_counter
        self.reminder_list[reminder_id] = {
            "reminder_id": reminder_id,
            "title": title,
            "description": description,
            "time": time,
            "notified": False,
        }
        return {"status": True, "message": f"提醒 '{title}' 已成功添加"}

    # 删除提醒
    def delete_reminder(self, reminder_id: int) -> Dict[str, Union[bool, str]]:
        """
        删除指定的提醒。
        Args:
            reminder_id (int): 要删除的提醒ID。
        Returns:
            Dict[str, Union[bool, str]]: 包含删除状态和结果的字典。
        """
        if self.logged_in == False:
            return {"status": False, "message": "device未登录，无法删除指定的提醒"}
        if reminder_id not in self.reminder_list:
            return {"status": False, "message": "提醒ID不存在"}

        del self.reminder_list[reminder_id]
        return {"status": True, "message": f"提醒ID {reminder_id} 已成功删除"}

    # 查看所有提醒
    def view_all_reminders(self) -> Dict[str, Union[bool, List[Dict[str, Union[str, datetime, bool]]]]]:
        """
        查看所有的提醒。
        Returns:
            Dict[str, Union[bool, List[Dict[str, Union[str, datetime, bool]]]]]: 包含所有提醒的字典列表。
        """
        if not self.reminder_list:
            return {"status": False, "message": "没有任何提醒"}

        reminders = []
        for reminder in self.reminder_list.values():
            reminders.append({
                "title": reminder["title"],
                "description": reminder["description"],
                "time": reminder["time"],
                "notified": reminder["notified"],
            })
        return {"status": True, "reminders": reminders}

    # 标记提醒为已通知
    def mark_as_notified(self, reminder_id: int) -> Dict[str, Union[bool, str]]:
        """
        标记提醒为已通知。
        Args:
            reminder_id (int): 要标记为已通知的提醒ID。
        Returns:
            Dict[str, Union[bool, str]]: 包含操作结果的字典。
        """
        if reminder_id not in self.reminder_list:
            return {"status": False, "message": "提醒ID不存在"}

        self.reminder_list[reminder_id]["notified"] = True
        return {"status": True, "message": f"提醒ID {reminder_id} 已标记为已通知"}

    # 搜索提醒
    def search_reminders(self, keyword: str) -> Dict[str, Union[bool, List[Dict[str, str]]]]:
        """
        根据关键词搜索提醒。
        Args:
            keyword (str): 搜索关键词。
        Returns:
            Dict[str, Union[bool, List[Dict[str, str]]]]: 包含匹配提醒的字典列表。
        """
        matched_reminders = []

        for reminder in self.reminder_list.values():
            if keyword.lower() in reminder["title"].lower() or keyword.lower() in reminder["description"].lower():
                matched_reminders.append({
                    "title": reminder["title"],
                    "description": reminder["description"],
                    "time": reminder["time"].strftime("%Y-%m-%d %H:%M"),
                })

        if not matched_reminders:
            return {"status": False, "message": "没有找到包含该关键词的提醒"}

        return {"status": True, "reminders": matched_reminders}