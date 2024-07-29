# Copyright (c) Alibaba, Inc. and its affiliates.

# Note: refer to https://github.com/hendrycks/test/blob/master/categories.py

subcategories = {
    'abstract_algebra': ['math'],
    'anatomy': ['health'],
    'astronomy': ['physics'],
    'business_ethics': ['business'],
    'clinical_knowledge': ['health'],
    'college_biology': ['biology'],
    'college_chemistry': ['chemistry'],
    'college_computer_science': ['computer science'],
    'college_mathematics': ['math'],
    'college_medicine': ['health'],
    'college_physics': ['physics'],
    'computer_security': ['computer science'],
    'conceptual_physics': ['physics'],
    'econometrics': ['economics'],
    'electrical_engineering': ['engineering'],
    'elementary_mathematics': ['math'],
    'formal_logic': ['philosophy'],
    'global_facts': ['other'],
    'high_school_biology': ['biology'],
    'high_school_chemistry': ['chemistry'],
    'high_school_computer_science': ['computer science'],
    'high_school_european_history': ['history'],
    'high_school_geography': ['geography'],
    'high_school_government_and_politics': ['politics'],
    'high_school_macroeconomics': ['economics'],
    'high_school_mathematics': ['math'],
    'high_school_microeconomics': ['economics'],
    'high_school_physics': ['physics'],
    'high_school_psychology': ['psychology'],
    'high_school_statistics': ['math'],
    'high_school_us_history': ['history'],
    'high_school_world_history': ['history'],
    'human_aging': ['health'],
    'human_sexuality': ['culture'],
    'international_law': ['law'],
    'jurisprudence': ['law'],
    'logical_fallacies': ['philosophy'],
    'machine_learning': ['computer science'],
    'management': ['business'],
    'marketing': ['business'],
    'medical_genetics': ['health'],
    'miscellaneous': ['other'],
    'moral_disputes': ['philosophy'],
    'moral_scenarios': ['philosophy'],
    'nutrition': ['health'],
    'philosophy': ['philosophy'],
    'prehistory': ['history'],
    'professional_accounting': ['other'],
    'professional_law': ['law'],
    'professional_medicine': ['health'],
    'professional_psychology': ['psychology'],
    'public_relations': ['politics'],
    'security_studies': ['politics'],
    'sociology': ['culture'],
    'us_foreign_policy': ['politics'],
    'virology': ['health'],
    'world_religions': ['philosophy'],
}

categories = {
    'STEM': ['physics', 'chemistry', 'biology', 'computer science', 'math', 'engineering'],
    'Humanities': ['history', 'philosophy', 'law'],
    'Social Science': ['politics', 'culture', 'economics', 'geography', 'psychology'],
    'Other': ['other', 'business', 'health'],
}


def main():

    reversed_categories = {}
    for category, subcategory_list in categories.items():
        for subcategory in subcategory_list:
            reversed_categories[subcategory] = category

    subject_mapping = {}
    for subject, subcategory_list in subcategories.items():
        category_name: str = reversed_categories[subcategory_list[0]]
        subject_show_name: str = ' '.join([item.capitalize() for item in subject.split('_')])
        subject_mapping[subject] = [subject_show_name, subcategory_list[0], category_name]

    print(subject_mapping)


if __name__ == '__main__':
    main()
