def convert_language(language: str) -> str:
    """Convert language names from BFCL v3 to BFCL v4 naming conventions."""
    from bfcl_eval.constants.enums import Language
    mapping = {
        'python': Language.PYTHON,
        'java': Language.JAVA,
        'javascript': Language.JAVASCRIPT,
    }
    return mapping[language.lower()]


def convert_format_language(format_language: str) -> str:
    """Convert format language names from BFCL v3 to BFCL v4 naming conventions."""
    from bfcl_eval.constants.enums import ReturnFormat
    mapping = {
        'python': ReturnFormat.PYTHON,
        'java': ReturnFormat.JAVA,
        'javascript': ReturnFormat.JAVASCRIPT,
        'json': ReturnFormat.JSON,
        'verbose_xml': ReturnFormat.VERBOSE_XML,
        'concise_xml': ReturnFormat.CONCISE_XML,
    }
    return mapping[format_language.lower()]
