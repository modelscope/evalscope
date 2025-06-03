import argparse


def add_argument(parser: argparse.ArgumentParser):
    parser.add_argument('--share', action='store_true', help='Share the app.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='The server name.')
    parser.add_argument('--server-port', type=int, default=None, help='The server port.')
    parser.add_argument('--debug', action='store_true', help='Debug the app.')
    parser.add_argument('--lang', type=str, default='zh', help='The locale.', choices=['zh', 'en'])
    parser.add_argument('--outputs', type=str, default='./outputs', help='The outputs dir.')
    parser.add_argument('--allowed-paths', nargs='+', default=['/'], help='The outputs dir.')
