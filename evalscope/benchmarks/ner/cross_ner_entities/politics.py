def get_entity_mappings():
    entity_type_map = {
        'COUNTRY': 'country',
        'ELECTION': 'election',
        'EVENT': 'event',
        'LOCATION': 'location',
        'MISC': 'misc',
        'ORGANIZATION': 'organisation',
        'PERSON': 'person',
        'POLITICAL_PARTY': 'political_party',
        'POLITICIAN': 'politician',
    }
    entity_descriptions = {
        'COUNTRY': ('A country or sovereign state (e.g., "United States", "Germany").'),
        'ELECTION': ('A specific election event (e.g., "2024 presidential election", "midterm elections").'),
        'EVENT': (
            'A significant political event, summit, or incident (e.g., "G7 Summit", "Brexit", "Watergate scandal").'
        ),
        'LOCATION': ('A politically significant building or location (e.g., "The White House", "10 Downing Street").'),
        'MISC': (
            'Miscellaneous political terms, ideologies, or documents (e.g., "democracy", '
            '"impeachment", "the Constitution").'
        ),
        'ORGANIZATION': (
            'A political or governmental organization (e.g., "United Nations", "NATO", "European Union").'
        ),
        'PERSON': (
            'A person mentioned in a political context who is not a politician (e.g., a journalist, an activist).'
        ),
        'POLITICAL_PARTY': ('A named political party (e.g., "Democratic Party", "Conservative Party").'),
        'POLITICIAN': ('A person who holds or seeks political office (e.g., "Joe Biden", "Angela Merkel").'),
    }
    return entity_type_map, entity_descriptions
