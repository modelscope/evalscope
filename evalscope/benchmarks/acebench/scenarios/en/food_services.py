
from datetime import datetime
from typing import Dict, List, Union

from .base_api import BaseApi


class FoodPlatform(BaseApi):
    """
    A class representing a Meituan-like platform where users need to log in, and each user has an associated balance.
    Merchants provide a menu of products, and users can place food delivery or group purchase orders.
    """

    def __init__(self):
        """
        Initialize the platform with a set of users, merchants, and their menus.
        """
        # Initialize users and their initial balances
        self.users: Dict[str, Dict[str, Union[str, float]]] = {
            "Eve": {"user_id": "U100", "password": "password123", "balance": 500.0},
            "Frank": {"user_id": "U101", "password": "password456", "balance": 300.0},
            "Grace": {"user_id": "U102", "password": "password789", "balance": 150.0},
            "Helen": {"user_id": "U103", "password": "password321", "balance": 800.0},
            "Isaac": {"user_id": "U104", "password": "password654", "balance": 400.0},
            "Jack": {"user_id": "U105", "password": "password654", "balance": 120.0},
        }

        # Initialize six merchants and their menus
        self.merchant_list: Dict[str, Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]] = {
            "Domino's": {
                "merchant_id": "M100",
                "service_type": "Pizza",
                "menu": [
                    {"product": "Margherita Pizza", "price": 68.0},
                    {"product": "Super Supreme Pizza", "price": 88.0},
                ],
            },
            "Rice Village Bibimbap": {
                "merchant_id": "M101",
                "service_type": "Bibimbap",
                "menu": [
                    {"product": "Stone Pot Bibimbap", "price": 35.0},
                    {"product": "Korean Beef Bibimbap", "price": 45.0},
                ],
            },
            "Haidilao": {
                "merchant_id": "M102",
                "service_type": "Hotpot",
                "menu": [
                    {"product": "Beef Rolls", "price": 68.0},
                    {"product": "Seafood Platter", "price": 88.0},
                ],
            },
            "Heytea": {
                "merchant_id": "M103",
                "service_type": "Milk Tea",
                "menu": [
                    {"product": "Cheese Milk Tea", "price": 25.0},
                    {"product": "Four Seasons Spring Milk Tea", "price": 22.0},
                ],
            },
            "Hema Fresh": {
                "merchant_id": "M104",
                "service_type": "Fresh Grocery",
                "menu": [
                    {"product": "Organic Vegetable Pack", "price": 15.0},
                    {"product": "Fresh Gift Pack", "price": 99.0},
                ],
            },
            "Jiutian BBQ": {
                "merchant_id": "M105",
                "service_type": "BBQ",
                "menu": [
                    {"product": "Korean Grilled Beef", "price": 128.0},
                    {"product": "Grilled Pork Belly", "price": 78.0},
                ],
            },
        }

        # Initialize list of logged-in users
        self.logged_in_users: List[str] = []

        # Initialize order history
        self.orders: List[Dict[str, Union[str, int, float, datetime]]] = []


    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)
        self.logged_in_users = scenario.get("logged_in_users",[])



    def login_food_platform(self, username: str, password: str) -> Dict[str, Union[bool, str]]:
        if self.wifi == False:
            return {"status": False, "message": "Wi-Fi is not enabled, unable to login"}
        if username not in self.users:
            return {"status": False, "message": "User does not exist"}
        if self.users[username]["password"] != password:
            return {"status": False, "message": "Incorrect password"}
        
        # Check if the user is already logged in
        if username in self.logged_in_users:
            return {"status": False, "message": f"{username} is already logged in"}
        
        # Record the logged-in user
        self.logged_in_users.append(username)
        return {"status": True, "message": f"User {username} has successfully logged in!"}


    def view_logged_in_users(self) -> Dict[str, Union[bool, List[str]]]:
        """
        View all currently logged-in users.
        Returns:
            Dict[str, Union[bool, List[str]]]: List of logged-in users.
        """
        if not self.logged_in_users:
            return {"status": False, "message": "No users are currently logged in to the food platform"}
        
        return {"status": True, "logged_in_users": self.logged_in_users}

    def check_balance(self, user_name: str) -> float:
        """
        Check the balance of the specified user.
        """
        if user_name in self.users:
            return self.users[user_name]["balance"]
        else:
            print(f"User {user_name} does not exist!")
            return 0.0



    def add_food_delivery_order(
        self, 
        username: str, 
        merchant_name: str, 
        items: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, Union[bool, str]]:
        if username not in self.logged_in_users:
            return {
                "status": False, 
                "message": f"User {username} is not logged in to the food platform"
            }

        if merchant_name not in self.merchant_list:
            return {"status": False, "message": "Merchant does not exist"}

        total_price = 0.0
        order_items = []

        for item in items:
            product_name = item.get("product")
            quantity = item.get("quantity", 1)

            if not isinstance(quantity, int) or quantity <= 0:
                return {
                    "status": False, 
                    "message": f"Invalid quantity {quantity} for product {product_name}"
                }

            # Find the product price
            product_found = False
            for product in self.merchant_list[merchant_name]["menu"]:
                if product["product"] == product_name:
                    total_price += product["price"] * quantity
                    order_items.append({
                        "product": product_name,
                        "quantity": quantity,
                        "price_per_unit": product["price"]
                    })
                    product_found = True
                    break
            if not product_found:
                return {
                    "status": False, 
                    "message": f"Product {product_name} does not exist in {merchant_name}'s menu"
                }

        # Check if the balance is sufficient
        if total_price > self.users[username]["balance"]:
            return {"status": False, "message": "Insufficient balance to place the order"}

        # Deduct the balance and create the order
        self.users[username]["balance"] -= total_price
        order = {
            "user_name": username,
            "merchant_name": merchant_name,
            "items": order_items,
            "total_price": total_price,
        }
        self.orders.append(order)
        return {
            "status": True, 
            "message": f"Food delivery order successfully placed with {merchant_name}. Total amount: {total_price} yuan"
        }

    
    def get_products(self, merchant_name: str) -> Union[List[Dict[str, Union[str, float]]], str]:
        """
        Retrieve the product list of a specific merchant.

        :param merchant_name: Name of the merchant.
        :return: Merchant's menu list or an error message.
        """
        merchant = self.merchant_list.get(merchant_name)
        if merchant:
            return merchant["menu"]
        else:
            return {
                "status": False, 
                "message": f"Merchant '{merchant_name}' does not exist"
            }
            

    def view_orders(self, user_name: str) -> Dict[str, Union[bool, str, List[Dict[str, Union[str, int, float]]]]]:
        """
        View all orders of a user.
        Args:
            user_name (str): Username.
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, Union[str, int, float]]]]]: List of user's orders.
        """
        user_orders = [order for order in self.orders if order['user_name'] == user_name]

        if not user_orders:
            return {"status": False, "message": "User has no order records"}
        
        return {"status": True, "orders": user_orders}


    def search_orders(self, keyword: str) -> Dict[str, Union[bool, str, List[Dict[str, Union[str, float]]]]]:
        """
        Search orders based on a keyword.
        Args:
            keyword (str): Keyword.
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, Union[str, float]]]]]: List of matching orders.
        """
        matched_orders = [
            order for order in self.orders 
            if keyword.lower() in order['merchant_name'].lower() 
            or any(keyword.lower() in item['product'].lower() for item in order.get('items', []))
        ]

        if not matched_orders:
            return {"status": False, "message": "No matching orders found"}

        return {"status": True, "orders": matched_orders}

