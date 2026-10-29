def get_entity_mappings():
    entity_type_map = {
        'ALBUM': 'album',
        'AWARD': 'award',
        'BAND': 'band',
        'COUNTRY': 'country',
        'EVENT': 'event',
        'LOCATION': 'location',
        'MISC': 'misc',
        'MUSICAL_ARTIST': 'musical_artist',
        'MUSICAL_INSTRUMENT': 'musical_instrument',
        'MUSIC_GENRE': 'music_genre',
        'ORGANIZATION': 'organisation',
        'PERSON': 'person',
        'SONG': 'song',
    }
    entity_descriptions = {
        'ALBUM': ('The title of a music album (e.g., "Abbey Road", "Thriller", "Lemonade").'),
        'AWARD': ('A music award or prize (e.g., "Grammy Award", "MTV Music Award").'),
        'BAND': ('The name of a musical group or band (e.g., "The Beatles", "Queen", "BTS").'),
        'COUNTRY': ('A country relevant to the music context (e.g., "USA", "UK", "South Korea").'),
        'EVENT': ('A music festival, concert tour, or event (e.g., "Glastonbury Festival", "Woodstock").'),
        'LOCATION': (
            'A venue, studio, or place relevant to music (e.g., "Madison Square Garden", "Abbey Road Studios").'
        ),
        'MISC': ('Miscellaneous music-related terms (e.g., "synthesizer", "major key", "a cappella").'),
        'MUSICAL_ARTIST': ('A solo musician or singer (e.g., "Michael Jackson", "Taylor Swift", "Ed Sheeran").'),
        'MUSICAL_INSTRUMENT': ('A musical instrument (e.g., "guitar", "piano", "violin").'),
        'MUSIC_GENRE': ('A genre or style of music (e.g., "Rock", "Pop", "Jazz", "K-Pop").'),
        'ORGANIZATION': ('A record label or music organization (e.g., "Capitol Records", "Sony Music").'),
        'PERSON': (
            'A person related to music who is not a primary artist (e.g., a producer, a songwriter, "John Lennon").'
        ),
        'SONG': ('The title of a song (e.g., "Bohemian Rhapsody", "Hey Jude", "Dynamite").'),
    }
    return entity_type_map, entity_descriptions
