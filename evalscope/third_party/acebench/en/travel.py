
from datetime import datetime, timedelta


class Travel:
    def __init__(self):
        """
        Initialize the system, including user profiles and flight information
        """
        # Initialize user information
        self.users = {
            "user1": {"user_name": "Eve", "password": "password123", "cash_balance": 2000.0, "bank_balance": 50000.0, "membership_level": "regular"},
            "user2": {"user_name": "Frank", "password": "password456", "cash_balance": 8000.0, "bank_balance": 8000.0, "membership_level": "silver"},
            "user3": {"user_name": "Grace", "password": "password789", "cash_balance": 1000.0, "bank_balance": 5000.0, "membership_level": "gold"}
        }

        # Initialize flight information
        self.flights = [
            {
                "flight_no": "CA1234",
                "origin": "Beijing",
                "destination": "Shanghai",
                "depart_time": "2024-07-15 08:00:00",
                "arrival_time": "2024-07-15 10:30:00",
                "status": "available",
                "seats_available": 5,
                "economy_price": 1200,
                "business_price": 3000
            },
            {
                "flight_no": "MU5678",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-16 09:00:00",
                "arrival_time": "2024-07-16 11:30:00",
                "status": "available",
                "seats_available": 3,
                "economy_price": 1900,
                "business_price": 3000
            },
            {
                "flight_no": "CZ4321",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-16 20:00:00",
                "arrival_time": "2024-07-16 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 2500,
                "business_price": 4000
            },
            {
                "flight_no": "CZ4352",
                "origin": "Shanghai",
                "destination": "Beijing",
                "depart_time": "2024-07-17 20:00:00",
                "arrival_time": "2024-07-17 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1600,
                "business_price": 2500
            },
            {
                "flight_no": "MU3561",
                "origin": "Beijing",
                "destination": "Nanjing",
                "depart_time": "2024-07-18 08:00:00",
                "arrival_time": "2024-07-18 10:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 4000
            },
            {
                "flight_no": "MU1566",
                "origin": "Beijing",
                "destination": "Nanjing",
                "depart_time": "2024-07-18 20:00:00",
                "arrival_time": "2024-07-18 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 4000
            },
            {
                "flight_no": "CZ1765",
                "origin": "Nanjing",
                "destination": "Shenzhen",
                "depart_time": "2024-07-17 20:30:00",
                "arrival_time": "2024-07-17 22:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "CZ1765",
                "origin": "Nanjing",
                "destination": "Shenzhen",
                "depart_time": "2024-07-18 12:30:00",
                "arrival_time": "2024-07-18 15:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "MH1765",
                "origin": "Xiamen",
                "destination": "Chengdu",
                "depart_time": "2024-07-17 12:30:00",
                "arrival_time": "2024-07-17 15:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "MH2616",
                "origin": "Chengdu",
                "destination": "Xiamen",
                "depart_time": "2024-07-18 18:30:00",
                "arrival_time": "2024-07-18 21:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            },
            {
                "flight_no": "MH2616",
                "origin": "Chengdu",
                "destination": "Fuzhou",
                "depart_time": "2024-07-16 18:30:00",
                "arrival_time": "2024-07-16 21:00:00",
                "status": "available",
                "seats_available": 8,
                "economy_price": 1500,
                "business_price": 2500
            }
        ]

        # Initialize reservation list
        self.reservations = [
            {
                "reservation_id": "res_1",
                "user_id": "user1",
                "flight_no": "CA1234",
                "payment_method": "bank",
                "cabin": "Economy Class",
                "baggage": 1,
                "origin": "Beijing",
                "destination": "Shanghai",
            },
            {
                "reservation_id": "res_2",
                "user_id": "user1",
                "flight_no": "MU5678",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Shanghai",
                "destination": "Beijing",
            },
            {
                "reservation_id": "res_3",
                "user_id": "user2",
                "flight_no": "MH1765",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Xiamen",
                "destination": "Chengdu",
            },
            {
                "reservation_id": "res_4",
                "user_id": "user2",
                "flight_no": "MU2616",
                "payment_method": "bank",
                "cabin": "Business Class",
                "baggage": 1,
                "origin": "Chengdu",
                "destination": "Xiamen",
            },
        ]


    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        pass
    def get_flight_details(self, origin: str = None, destination: str = None) -> list:
        """
        Query flight basic information based on origin and destination.
        """
        flights = self.flights
        
        if origin:
            flights = [flight for flight in flights if flight["origin"] == origin]
        
        if destination:
            flights = [flight for flight in flights if flight["destination"] == destination]
        if len(flights) == 0:
            return f'There are no direct flights that meet the criteria.'

        return [{"flight_no": flight["flight_no"], "origin": flight["origin"], "destination": flight["destination"], 
                "depart_time": flight["depart_time"], "arrival_time": flight["arrival_time"], 
                "status": flight["status"], "seats_available": flight["seats_available"], 
                "economy_price": flight["economy_price"], "business_price": flight["business_price"]}
                for flight in flights]


    def get_user_details(self, user_id: str, password: str) -> dict:
        """
        Query user information based on username and password
        """
        user = self.users.get(user_id)
        if user and user["password"] == password:
            return {key: value for key, value in user.items() if key != "password"}
        return {"status": "error", "message": "Incorrect username or password."}
    

    def get_reservation_details(self, reservation_id: str = None, user_id: str = None) -> list:
        """
        Query reservation information based on reservation ID or user ID, including basic flight information.
        """
        # Filter reservations by reservation ID or user ID
        if reservation_id:
            reservations = [reservation for reservation in self.reservations if reservation["reservation_id"] == reservation_id]
        elif user_id:
            reservations = [reservation for reservation in self.reservations if reservation["user_id"] == user_id]
        else:
            return {"status": "error", "message": "Please provide a valid reservation ID or user ID"}

        # Add flight information for each reservation
        detailed_reservations = []
        for reservation in reservations:
            flight_info = next((flight for flight in self.flights if flight["flight_no"] == reservation["flight_no"]), None)
            detailed_reservation = {**reservation, "flight_info": flight_info}
            detailed_reservations.append(detailed_reservation)
        
        return detailed_reservations
    

    
    
    def authenticate_user(self, user_id, password):
        user = self.users.get(user_id)
        if user and user["password"] == password:
            return user
        return {"status": "error", "message": "Incorrect username or password."}

    

    def get_baggage_allowance(self, membership_level, cabin_class):
        """
        Get free checked baggage allowance based on membership level and cabin class.

        Parameters:
        - membership_level (str): Membership level ("regular", "silver", "gold")
        - cabin_class (str): Cabin class ("Economy Class", "Business Class")

        Returns:
        - int: Number of free checked baggage
        """
        allowance = {
            "regular": { "Economy Class": 1, "Business Class": 2},
            "silver": { "Economy Class": 2, "Business Class": 3},
            "gold": {"Economy Class": 3, "Business Class": 3}
        }
        return allowance.get(membership_level, {}).get(cabin_class, 0)
    
    # 查询中转航班
    def find_transfer_flights(self, origin_city, transfer_city, destination_city):
        """
        Find connecting flights from origin city to destination city via transfer city,
        ensuring the first flight's arrival time is earlier than the second flight's departure time.
        :param origin_city: Departure city
        :param transfer_city: Transfer city
        :param destination_city: Arrival city
        :return: List of connecting flights that meet the criteria, each containing information for both flight segments.
        """
        # Get flights from departure city to transfer city
        first_leg_flights = [
            flight for flight in self.flights 
            if flight["origin"] == origin_city and flight["destination"] == transfer_city and flight["status"] == "available"
        ]

        # Get flights from transfer city to destination city
        second_leg_flights = [
            flight for flight in self.flights 
            if flight["origin"] == transfer_city and flight["destination"] == destination_city and flight["status"] == "available"
        ]

        # Store valid transfer flight combinations
        transfer_flights = []

        # Check each combination of first and second leg flights for valid connections
        for first_flight in first_leg_flights:
            first_arrival = datetime.strptime(first_flight["arrival_time"], "%Y-%m-%d %H:%M:%S")
            
            for second_flight in second_leg_flights:
                second_departure = datetime.strptime(second_flight["depart_time"], "%Y-%m-%d %H:%M:%S")
                
                # Check if first flight arrives before second flight departs
                if first_arrival < second_departure:
                    transfer_flights.append({
                        "first_leg": first_flight,
                        "second_leg": second_flight
                    })

        # Return list of valid transfer flights
        if transfer_flights:
            return transfer_flights
        else:
            return "No connecting flights that meet the criteria were found."
    

    def calculate_baggage_fee(self, membership_level, cabin_class, baggage_count):
        free_baggage = {"regular": {"Economy Class": 1, "Business Class": 2}, "silver": {"Economy Class": 2, "Business Class": 3}, "gold": {"Economy Class": 3, "Business Class": 3}}
        free_limit = free_baggage[membership_level][cabin_class]
        additional_baggage = max(baggage_count - free_limit, 0)
        return additional_baggage * 50
    

    def update_balance(self, user, payment_method, amount):
        """
        Update user's balance.
        :param user: User information
        :param payment_method: Payment method ("cash" or "bank")
        :param amount: Update amount (positive for increase, negative for decrease)
        :return: Returns True if balance is sufficient and update successful, False otherwise.
        """
        if payment_method == "cash":
            if user["cash_balance"] + amount < 0:
                return False  # Insufficient balance
            user["cash_balance"] += amount
        elif payment_method == "bank":
            if user["bank_balance"] + amount < 0:
                return False  # Insufficient balance
            user["bank_balance"] += amount
        return True
    
    # 预定航班
    def reserve_flight(self, user_id, password, flight_no, cabin, payment_method, baggage_count):
        user = self.authenticate_user(user_id, password)
        if not user:
            return "Authentication failed. Please check your user ID and password."

        # Check flight and seats availability
        flight = next((f for f in self.flights if f["flight_no"] == flight_no and f["status"] == "available"), None)

        # Calculate flight price
        price = flight["economy_price"] if cabin == "Economy Class" else flight["business_price"]
        total_cost = price

        # Calculate baggage fee
        baggage_fee = self.calculate_baggage_fee(user["membership_level"], cabin, baggage_count)
        total_cost += baggage_fee

        # Check payment method
        if payment_method not in ["cash", "bank"]:
            return "Invalid payment method"
            
        # Update balance after reservation
        if payment_method == "cash":
            if total_cost > self.users.get(user_id)["cash_balance"]:
                return f"Your cash balance is insufficient. Please consider using another payment method."
            self.users.get(user_id)["cash_balance"] -= total_cost
        else:
            if total_cost > self.users.get(user_id)["bank_balance"]:
                return f"Your bank balance is insufficient. Please consider using another payment method."
            self.users.get(user_id)["bank_balance"] -= total_cost

        # Update flight information and generate reservation
        flight["seats_available"] -= 1
        reservation_id = f"res_{len(self.reservations) + 1}"
        reservation = {
            "reservation_id": reservation_id,
            "user_id": user_id,
            "flight_no": flight_no,
            "payment_method": payment_method,
            "cabin": cabin,
            "baggage": baggage_count,
        }
        self.reservations.append(reservation)

        return f"Booking successful. Reservation ID: {reservation_id}. Total cost: {total_cost} yuan (including baggage fees)."


    def modify_flight(self, user_id, reservation_id, new_flight_no=None, new_cabin=None, add_baggage=0, new_payment_method=None):
        # Get corresponding reservation
        reservation = next((r for r in self.reservations if r['reservation_id'] == reservation_id and r['user_id'] == user_id), None)
        if not reservation:
            return "Reservation not found or user ID does not match."

        # Check current flight information
        current_flight = next((f for f in self.flights if f['flight_no'] == reservation['flight_no']), None)
        if not current_flight:
            return "Flight information not found."

        # Get original payment method or new provided payment method
        payment_method = new_payment_method if new_payment_method else reservation['payment_method']
        user = self.users[user_id]
        if not user:
            return "User information not found."

        # Store processing results
        result_messages = []

        # Update flight number if provided and matches origin/destination
        if new_flight_no and new_flight_no != reservation['flight_no']:
            new_flight = next((f for f in self.flights if f['flight_no'] == new_flight_no), None)
            if new_flight and new_flight['origin'] == current_flight['origin'] and new_flight['destination'] == current_flight['destination']:
                reservation['flight_no'] = new_flight_no
                result_messages.append("Flight number has been changed.")
            else:
                return f"Flight change failed: Invalid new flight number or destination does not match."

        # Update cabin if provided and calculate price difference
        if new_cabin and new_cabin != reservation.get('cabin'):
            price_difference = self.calculate_price_difference(current_flight, reservation['cabin'], new_cabin)
            reservation['cabin'] = new_cabin
            if price_difference > 0:
                # Deduct price difference
                if self.update_balance(user, payment_method, -price_difference):
                    result_messages.append(f"Cabin change successful. Price difference paid: {price_difference}.")
                else:
                    result_messages.append("Insufficient balance to pay the cabin price difference.")
            elif price_difference < 0:
                # Refund
                self.update_balance(user, payment_method, -price_difference)
                result_messages.append(f"Cabin change successful. Price difference refunded: {-price_difference}.")

        # Add checked baggage, check free allowance and calculate fees
        if add_baggage > 0:
            membership = user["membership_level"]
            max_free_baggage = self.get_baggage_allowance(membership, reservation['cabin'])
            current_baggage = reservation.get('baggage', 0)
            total_baggage = current_baggage + add_baggage
            extra_baggage = max(0, total_baggage - max_free_baggage)
            baggage_cost = extra_baggage * 50
            if baggage_cost > 0:
                # Deduct baggage fees
                if self.update_balance(user, payment_method, -baggage_cost):
                    result_messages.append(f"Baggage has been added. Additional fee to be paid: {baggage_cost}.")
                else:
                    result_messages.append("Insufficient balance to pay the additional baggage fees.")
            reservation['baggage'] = total_baggage

        # Return final result
        if not result_messages:
            result_messages.append("Modification completed with no additional fees.")
        return " ".join(result_messages)



    def cancel_reservation(self, user_id, reservation_id, reason):
            # Set the default current time to July 14, 2024, 6:00 AM
            current_time = datetime(2024, 7, 14, 6, 0, 0)

            # Verify that the user and reservation exist
            user = self.users.get(user_id, None)
            if not user:
                return "Invalid user ID."

            reservation = next((r for r in self.reservations if r["reservation_id"] == reservation_id and r["user_id"] == user_id), None)
            if not reservation:
                return "Invalid reservation ID or it does not belong to the user."

            # Check if flight information exists
            flight = next((f for f in self.flights if f["flight_no"] == reservation["flight_no"]), None)
            if not flight:
                return "Invalid flight information."
            
            # Check if the flight has already departed
            depart_time = datetime.strptime(flight["depart_time"], "%Y-%m-%d %H:%M:%S")
            if current_time > depart_time:
                return "The flight segment has been used and cannot be canceled."

            # Calculate the time until departure
            time_until_departure = depart_time - current_time
            cancel_fee = 0
            refund_amount = 0

            # Get the flight price
            flight_price = flight["economy_price"] if reservation["cabin"] == "Economy Class" else flight["business_price"]

            # Cancellation policy and refund calculation
            if reason == "The airline has canceled the flight.":
                # Airline cancels the flight, full refund
                refund_amount = flight_price
                self.process_refund(user, refund_amount)
                return f"The flight has been canceled. Your reservation will be canceled free of charge, and {refund_amount} yuan has been refunded."

            elif time_until_departure > timedelta(days=1):
                # More than 24 hours before departure, free cancellation
                refund_amount = flight_price
                self.process_refund(user, refund_amount)
                return f"More than 24 hours before departure. Free cancellation successful, {refund_amount} yuan has been refunded."
            
            else:
                # If not eligible for free cancellation, set a cancellation fee as needed
                cancel_fee = flight_price * 0.1  # Assume a cancellation fee of 10% of the ticket price
                refund_amount = flight_price - cancel_fee
                self.process_refund(user, refund_amount)
                return f"Less than 24 hours before departure. A cancellation fee of {cancel_fee} yuan has been deducted, and {refund_amount} yuan has been refunded."



    def process_refund(self, user, amount):
        """
        Add refund amount to user's cash balance.
        """
        user["cash_balance"] += amount
        print(f"Refund has been successfully processed. {user['user_name']}'s cash balance has been increased by {amount} yuan.")


    def calculate_price_difference(self, flight, old_cabin, new_cabin):
        """
        Calculate the price difference between different cabin classes
        """
        cabin_prices = {
            "Economy Class": flight["economy_price"],
            "Business Class": flight["business_price"]
        }
        old_price = cabin_prices.get(old_cabin, 0)
        new_price = cabin_prices.get(new_cabin, 0)
        return new_price - old_price


