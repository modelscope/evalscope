def get_entity_mappings():
    entity_type_map = {
        'ACADEMICJOURNAL': 'academic_journal',
        'ASTRONOMICALOBJECT': 'astronomical_object',
        'AWARD': 'award',
        'CHEMICALCOMPOUND': 'chemical_compound',
        'CHEMICALELEMENT': 'chemical_element',
        'COUNTRY': 'country',
        'DISCIPLINE': 'discipline',
        'ENZYME': 'enzyme',
        'EVENT': 'event',
        'LOCATION': 'location',
        'MISC': 'misc',
        'ORGANISATION': 'organisation',
        'PERSON': 'person',
        'PROTEIN': 'protein',
        'SCIENTIST': 'scientist',
        'THEORY': 'theory',
        'UNIVERSITY': 'university'
    }
    entity_descriptions = {
        'ACADEMICJOURNAL': ('A scientific journal or publication (e.g., "Nature", "Science", "The Lancet").'),
        'ASTRONOMICALOBJECT': ('A natural object in space (e.g., "Mars", "Andromeda Galaxy", '
                               '"Halley\'s Comet").'),
        'AWARD': ('A scientific award or prize (e.g., "Nobel Prize in Physics", "Fields Medal").'),
        'CHEMICALCOMPOUND':
        ('A chemical substance consisting of two or more elements (e.g., "H2O", '
         '"Carbon Dioxide").'),
        'CHEMICALELEMENT': ('An element from the periodic table (e.g., "Hydrogen", "Oxygen", "Gold").'),
        'COUNTRY': ('A country relevant to a scientific context (e.g., "Switzerland" for CERN).'),
        'DISCIPLINE':
        ('A branch of science or academic discipline (e.g., "Physics", '
         '"Molecular Biology", "Astronomy").'),
        'ENZYME': ('A specific type of protein that acts as a catalyst (e.g., "Lactase", "Catalase").'),
        'EVENT': ('A significant scientific mission or event (e.g., "Apollo 11 mission", '
                  '"Human Genome Project").'),
        'LOCATION':
        ('A research facility or location of scientific importance (e.g., "CERN", '
         '"International Space Station").'),
        'MISC':
        ('Miscellaneous scientific terms or concepts (e.g., "double helix", '
         '"black hole", "quantum mechanics").'),
        'ORGANISATION': ('A scientific organization or agency (e.g., "NASA", "Max Planck Society", "WHO").'),
        'PERSON':
        ('A person mentioned in a scientific context who is not a scientist '
         '(e.g., a patient, a benefactor).'),
        'PROTEIN': ('A specific protein (that is not an enzyme) (e.g., "Hemoglobin", '
                    '"Insulin", "Keratin").'),
        'SCIENTIST':
        ('A person who is a scientist, researcher, or inventor (e.g., "Albert Einstein", '
         '"Marie Curie").'),
        'THEORY': ('A named scientific theory or law (e.g., "Theory of Relativity", '
                   '"Big Bang Theory").'),
        'UNIVERSITY':
        ('A university or academic institution involved in science (e.g., '
         '"Cambridge University", "Caltech").')
    }
    return entity_type_map, entity_descriptions
