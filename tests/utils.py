import os

TEST_LEVEL_LIST = [0, 1]
# Example: export TEST_LEVEL_LIST=0,1
TEST_LEVEL_LIST_STR = 'TEST_LEVEL_LIST'


def test_level_list():
    global TEST_LEVEL_LIST
    if TEST_LEVEL_LIST_STR in os.environ:
        TEST_LEVEL_LIST = [int(x) for x in os.environ[TEST_LEVEL_LIST_STR].split(',')]

    return TEST_LEVEL_LIST
