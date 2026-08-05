
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
                "description": """First, help Frank place a food delivery order at "Hema Fresh," ordering two "Fresh Gift Packs." Then, send a message to Frank saying, "The price of the purchased goods is () yuan." Replace the parentheses with the actual amount, keeping one decimal place.""",
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
        Check if the reminder capacity is full.
        Returns:
            bool: Returns True if capacity is full, False otherwise.
        """
        return len(self.reminder_list) >= self.max_capacity
    
    def view_reminder_by_title(self, title: str) -> Dict[str, Union[str, bool, Dict[str, Union[str, bool, datetime]]]]:
        """
        View a specific reminder by its title.
        Args:
            title (str): The title of the reminder.
        Returns:
            Dict[str, Union[str, bool, Dict]]: A dictionary containing the search status and reminder details.
        """
        if self.logged_in == False:
            return {"status": False, "message": "The device is not logged in, so you cannot view notifications"}
        for reminder_id, reminder in self.reminder_list.items():
            if reminder["title"] == title:
                return {"status": True, "reminder": reminder}
        
        return {"status": False, "message": f"No reminder found with the title '{title}'."}


    def add_reminder(self, title: str, description: str, time: datetime) -> Dict[str, Union[bool, str]]:
        """
        Add a new reminder.
        Args:
            title (str): Reminder title.
            description (str): Reminder description.
            time (datetime): Reminder time.
        Returns:
            Dict[str, Union[bool, str]]: A dictionary containing the addition status and result.
        """

        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in. Unable to add a new reminder."}
        if self._check_capacity():
            return {"status": False, "message": "Reminder capacity is full. Unable to add a new reminder."}


        self.reminder_id_counter += 1
        reminder_id = self.reminder_id_counter
        self.reminder_list[reminder_id] = {
            "reminder_id": reminder_id,
            "title": title,
            "description": description,
            "time": time,
            "notified": False,
        }
        return {"status": True, "message": f"Reminder '{title}' was successfully added."}


    # 2. Delete Reminder
    def delete_reminder(self, reminder_id: int) -> Dict[str, Union[bool, str]]:
        """
        Deletes the specified reminder.
        Args:
            reminder_id (int): The ID of the reminder to delete.
        Returns:
            Dict[str, Union[bool, str]]: A dictionary containing the deletion status and result.
        """
        if self.logged_in == False:
            return {"status": False, "message": "Device not logged in. Unable to delete the specified reminder."}
        if reminder_id not in self.reminder_list:
            return {"status": False, "message": "Reminder ID does not exist."}

        del self.reminder_list[reminder_id]
        return {"status": True, "message": f"Reminder ID {reminder_id} was successfully deleted."}

    # 3. View All Reminders
    def view_all_reminders(self) -> Dict[str, Union[bool, List[Dict[str, Union[str, datetime, bool]]]]]:
        """
        Views all reminders.
        Returns:
            Dict[str, Union[bool, List[Dict[str, Union[str, datetime, bool]]]]]: A dictionary containing a list of all reminders.
        """
        if not self.reminder_list:
            return {"status": False, "message": "No reminders found."}

        reminders = []
        for reminder in self.reminder_list.values():
            reminders.append({
                "title": reminder["title"],
                "description": reminder["description"],
                "time": reminder["time"],
                "notified": reminder["notified"],
            })
        return {"status": True, "reminders": reminders}



 