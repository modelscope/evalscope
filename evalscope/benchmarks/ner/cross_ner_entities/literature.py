def get_entity_mappings():
    entity_type_map = {
        'AWARD': 'award',
        'BOOK': 'book',
        'COUNTRY': 'country',
        'EVENT': 'event',
        'LITERARYGENRE': 'literary_genre',
        'LOCATION': 'location',
        'MAGAZINE': 'magazine',
        'MISC': 'misc',
        'ORGANISATION': 'organisation',
        'PERSON': 'person',
        'POEM': 'poem',
        'WRITER': 'writer'
    }
    entity_descriptions = {
        'AWARD': ('A literary award or prize (e.g., "Nobel Prize in Literature", "Booker Prize").'),
        'BOOK': ('The title of a book (e.g., "Pride and Prejudice", "One Hundred Years of Solitude").'),
        'COUNTRY': ('A country relevant to the literary context (e.g., "England", "Russia").'),
        'EVENT': ('A literary festival or significant event (e.g., "Hay Festival", "Frankfurt Book Fair").'),
        'LITERARYGENRE':
        ('A genre or category of literature (e.g., "Science Fiction", "Gothic novel", '
         '"magical realism").'),
        'LOCATION': ('A real or fictional place mentioned in a literary context (e.g., "London", '
                     '"Middle-earth").'),
        'MAGAZINE': ('A magazine or literary journal (e.g., "The New Yorker", "Paris Review").'),
        'MISC': ('Miscellaneous literary terms (e.g., "protagonist", "sonnet", '
                 '"Shakespeare\'s Globe").'),
        'ORGANISATION': ('A publishing house or literary organization (e.g., "Penguin Random House").'),
        'PERSON': ('A character or person mentioned who is not a writer (e.g., "Elizabeth Bennet", '
                   '"King Lear").'),
        'POEM': ('The title of a poem (e.g., "The Waste Land", "Ozymandias").'),
        'WRITER': ('The name of a writer, author, or poet (e.g., "Jane Austen", '
                   '"Gabriel Garcia Marquez").')
    }
    return entity_type_map, entity_descriptions
