def get_entity_mappings():
    entity_type_map = {
        'ALGORITHM': 'algorithm',
        'CONFERENCE': 'conference',
        'COUNTRY': 'country',
        'FIELD': 'field',
        'LOCATION': 'location',
        'METRICS': 'metrics',
        'MISC': 'misc',
        'ORGANISATION': 'organisation',
        'PERSON': 'person',
        'PRODUCT': 'product',
        'PROGRAMLANG': 'programming_language',
        'RESEARCHER': 'researcher',
        'TASK': 'task',
        'UNIVERSITY': 'university'
    }
    entity_descriptions = {
        'ALGORITHM':
        ('A specific algorithm or model architecture in AI (e.g., "Transformer", '
         '"gradient descent", "ResNet").'),
        'CONFERENCE': ('An academic conference related to AI (e.g., "NeurIPS", "ICML", "CVPR").'),
        'COUNTRY': ('A country mentioned in the context of AI research or development '
                    '(e.g., "USA", "China").'),
        'FIELD':
        ('A sub-field or area of study within AI (e.g., "Natural Language Processing", '
         '"Computer Vision").'),
        'LOCATION':
        ('A specific geographical location relevant to AI, other than countries '
         '(e.g., "Silicon Valley").'),
        'METRICS': ('A performance metric used to evaluate AI models (e.g., "F1-score", '
                    '"BLEU", "accuracy").'),
        'MISC': ('Miscellaneous AI-related terms that don\'t fit other categories '
                 '(e.g., "Turing Award").'),
        'ORGANISATION':
        ('An organization, company, or lab involved in AI (e.g., "Google AI", '
         '"OpenAI", "DeepMind").'),
        'PERSON':
        ('A person mentioned in the context of AI, who is not a researcher '
         '(e.g., a CEO or public figure).'),
        'PRODUCT': ('An AI-related product, framework, or software (e.g., "TensorFlow", '
                    '"PyTorch", "AlphaGo").'),
        'PROGRAMLANG': ('A programming language used in AI (e.g., "Python", "C++", "Julia").'),
        'RESEARCHER': ('A person who conducts research in the field of AI (e.g., "Yann LeCun", '
                       '"Geoffrey Hinton").'),
        'TASK': (
            'A specific problem or task that AI is used to solve (e.g., "Image Classification", '
            '"Sentiment Analysis").'
        ),
        'UNIVERSITY':
        ('A university or academic institution involved in AI research (e.g., '
         '"Stanford University", "MIT").')
    }
    return entity_type_map, entity_descriptions
