
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
        # 设置用户和初始金额
        self.users: Dict[str, Dict[str, Union[str, float]]] = {
            "Eve": {"user_id": "U100", "password": "password123", "balance": 500.0},
            "Frank": {"user_id": "U101", "password": "password456", "balance": 300.0},
            "Grace": {"user_id": "U102", "password": "password789", "balance": 150.0},
            "Helen": {"user_id": "U103", "password": "password321", "balance": 800.0},
            "Isaac": {"user_id": "U104", "password": "password654", "balance": 400.0},
            "Jack": {"user_id": "U105", "password": "password654", "balance": 120.0},
        }

        # 设置六个商家及其菜单
        self.merchant_list: Dict[str, Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]] = {
            "达美乐": {
                "merchant_id": "M100",
                "service_type": "Pizza",
                "menu": [
                    {"product": "玛格丽特披萨", "price": 68.0},
                    {"product": "超级至尊披萨", "price": 88.0},
                ],
            },
            "米村拌饭": {
                "merchant_id": "M101",
                "service_type": "Bibimbap",
                "menu": [
                    {"product": "石锅拌饭", "price": 35.0},
                    {"product": "韩式牛肉拌饭", "price": 45.0},
                ],
            },
            "海底捞": {
                "merchant_id": "M102",
                "service_type": "Hotpot",
                "menu": [
                    {"product": "牛肉卷", "price": 68.0},
                    {"product": "海鲜拼盘", "price": 88.0},
                ],
            },
            "喜茶": {
                "merchant_id": "M103",
                "service_type": "Milk Tea",
                "menu": [
                    {"product": "芝士奶茶", "price": 25.0},
                    {"product": "四季春奶茶", "price": 22.0},
                ],
            },
            "盒马生鲜": {
                "merchant_id": "M104",
                "service_type": "Fresh Grocery",
                "menu": [
                    {"product": "有机蔬菜包", "price": 15.0},
                    {"product": "生鲜大礼包", "price": 99.0},
                ],
            },
            "九田家烤肉": {
                "merchant_id": "M105",
                "service_type": "BBQ",
                "menu": [
                    {"product": "韩式烤牛肉", "price": 128.0},
                    {"product": "烤五花肉", "price": 78.0},
                ],
            },
        }

        # 设置已登录用户列表
        self.logged_in_users: List[str] = []
        # 订单列表
        self.orders: List[Dict[str, Union[str, int, float, datetime]]] = []


    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)
        self.logged_in_users = scenario.get("logged_in_users",[])

       # 用户登录功能
    def login_food_platform(self, username: str, password: str) -> Dict[str, Union[bool, str]]:
        if self.wifi == False:
            return {"status": False, "message": "wifi未打开，无法登录"}
        if username not in self.users:
            return {"status": False, "message": "用户不存在"}
        if self.users[username]["password"] != password:
            return {"status": False, "message": "密码错误"}
        
        # 检查是否已经有用户登录
        if username in self.logged_in_users:
            return {"status": False, "message": f"{username} 已经登录"}

        # 记录已登录用户
        self.logged_in_users.append(username)
        return {"status": True, "message": f"用户{username}登陆成功！"}

    # 查看已登录用户列表
    def view_logged_in_users(self) -> Dict[str, Union[bool, List[str]]]:
        """
        查看当前所有已登录的用户。
        Returns:
            Dict[str, Union[bool, List[str]]]: 登录用户列表。
        """
        if not self.logged_in_users:
            return {"status": False, "message": "当前没有登录food platform"}
        
        return {"status": True, "logged_in_users": self.logged_in_users}

    def check_balance(self, user_name: str) -> float:
        """
        查询指定用户的余额。
        """
        if user_name in self.users:
            return self.users[user_name]["balance"]
        else:
            print(f"用户 {user_name} 不存在!")
            return 0.0


    # 添加外卖订单
    def add_food_delivery_order(
        self, 
        username: str, 
        merchant_name: str, 
        items: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, Union[bool, str]]:
        if username not in self.logged_in_users:
            return {
                "status": False, 
                "message": f"用户 {username} 未登录food platform"
            }

        if merchant_name not in self.merchant_list:
            return {"status": False, "message": "商家不存在"}

        total_price = 0.0
        order_items = []

        for item in items:
            product_name = item.get("product")
            quantity = item.get("quantity", 1)

            if not isinstance(quantity, int) or quantity <= 0:
                return {
                    "status": False, 
                    "message": f"无效的数量 {quantity} 对于商品 {product_name}"
                }

            # 查找商品价格
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
                    "message": f"商品 {product_name} 不存在于 {merchant_name} 的菜单中"
                }

        # 检查余额是否足够
        if total_price >= self.users[username]["balance"]:
            return {"status": False, "message": "余额不足，无法下单"}

        # 扣除余额并创建订单
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
            "message": f"外卖订单成功下单给 {merchant_name}，总金额为 {total_price} 元"
        }

    
    def get_products(self, merchant_name: str) -> Union[List[Dict[str, Union[str, float]]], str]:
        """
        查询特定商家的商品列表。

        :param merchant_name: 商家的名称。
        :return: 商家的菜单列表或错误消息。
        """
        merchant = self.merchant_list.get(merchant_name)
        if merchant:
            return merchant["menu"]
        else:
            return {
            "status": False, 
            "message": f"商家 '{merchant_name}' 不存在"
        }
        
        
    #  查看订单
    def view_orders(self, user_name: str) -> Dict[str, Union[bool, str, List[Dict[str, Union[str, int, float]]]]]:
        """
        查看用户的所有订单。
        Args:
            user_name (str): 用户名。
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, Union[str, int, float]]]]]: 用户的订单列表。
        """
        user_orders = [order for order in self.orders if order['user_name'] == user_name]

        if not user_orders:
            return {"status": False, "message": "用户没有订单记录"}
        
        return {"status": True, "orders": user_orders}

    # 搜索订单
    def search_orders(self, keyword: str) -> Dict[str, Union[bool, str, List[Dict[str, Union[str, float]]]]]:
        """
        根据关键词搜索订单。
        Args:
            keyword (str): 关键词。
        Returns:
            Dict[str, Union[bool, str, List[Dict[str, Union[str, float]]]]]: 匹配的订单列表。
        """
        matched_orders = [order for order in self.orders if keyword.lower() in order['merchant_name'].lower() or any(keyword.lower() in item.lower() for item in order.get('items', []))]

        if not matched_orders:
            return {"status": False, "message": "没有找到匹配的订单"}

        return {"status": True, "orders": matched_orders}

