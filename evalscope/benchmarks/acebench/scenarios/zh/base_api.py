
class BaseApi:

    def __init__(self):
        self.wifi = False
        self.logged_in = True
    
    def _load_scenario(self, scenario: dict, long_context=False) -> None:

        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)

    
    def turn_on_wifi(self):
        self.wifi = True
        return {'status': True, 'message': 'wifi已经打开'} 
    
    def login_device(self):
        self.logged_in = True
        return {'status': True, 'message': '设备已经登录'} 
        


