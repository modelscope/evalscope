from typing import Awaitable, Callable

hotel_data = [
    {
        'name': 'Grand Plaza Hotel',
        'location': 'Manchester',
        'amenities': ['Free WiFi', 'Gym', 'Spa'],
    },
    {
        'name': 'Oceanview Resort',
        'location': 'London',
        'amenities': ['Beach Access', 'Pool', 'Restaurant'],
    },
    {
        'name': 'Mountain Lodge',
        'location': 'Hull',
        'amenities': ['Skiing', 'Hot Tub', 'Free Breakfast'],
    },
    {
        'name': 'City Central Inn',
        'location': 'Southampton',
        'amenities': ['Conference Rooms', 'Airport Shuttle', 'Fitness Center'],
    },
    {
        'name': 'Grand Hotel',
        'location': 'Newcastle',
        'amenities': ['Free WiFi', 'Gym', 'Spa', 'Pool'],
    },
]


def search_hotels() -> Callable[..., Awaitable[str]]:
    """Build the official simulated search_hotels tool."""

    async def run(location: str, check_in_date: str, check_out_date: str) -> str:
        """
        Search for hotels based on location and check-in/check-out dates.

        Args:
            location: The location to search for hotels
            check_in_date: The check-in date
            check_out_date: The check-out date

        Returns:
            A list of hotels matching the search request.
        """
        if location:
            matching_hotels = [hotel for hotel in hotel_data if str(hotel['location']).lower() == location.lower()]
            if matching_hotels:
                return str(matching_hotels)
            else:
                return 'No matching hotels found'
        else:
            return 'Required argument missing for search_hotels: location'

    return run
