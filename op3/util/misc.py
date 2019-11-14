import os


def get_module_path():
    return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))