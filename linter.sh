yapf -r -i evalscope/ setup.py
isort -rc evalscope/ setup.py
flake8 evalscope/ setup.py
