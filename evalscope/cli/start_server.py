# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

from evalscope.cli.base import CLICommand

current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)
root_path = os.path.dirname(current_path)
print(root_path)


def subparser_func(args):
    """ Function which will be called for a specific sub parser.
    """
    return PerfServerCMD(args)


def add_perf_args(parser):
    parser.add_argument('--server-command', required=True, type=str, help='The start server command.')
    parser.add_argument(
        '--logdir',
        required=True,
        type=str,
        help='The monitor log save dir, tensorboard start at this path for display!')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The tensorboard host')
    parser.add_argument('--tensorboard-port', type=str, default='6006', help='The tensorboard port')


def async_run_command_with_popen(cmd):
    sub_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, encoding='utf8')
    return sub_process


def start_monitor(args):
    cmd = ['python', '%s/perf/monitor.py' % root_path, '--logdir', args.logdir]
    print(cmd)
    p = async_run_command_with_popen(cmd)
    os.set_blocking(p.stdout.fileno(), False)
    return p


def start_tensorboard(args):
    cmd = ['tensorboard', '--logdir', args.logdir, '--host', args.host, '--port', args.tensorboard_port]
    p = async_run_command_with_popen(cmd)
    os.set_blocking(p.stdout.fileno(), False)
    return p


def start_server(args):
    cmd = args.server_command
    print(cmd)
    sub_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        shell=True,
        universal_newlines=True,
        encoding='utf8')

    os.set_blocking(sub_process.stdout.fileno(), False)
    return sub_process


def wait_for_workers(workers):
    while True:
        for idx, worker in enumerate(workers):
            if worker is None:
                continue
            # check worker is completed.
            if worker.poll() is None:
                for line in iter(worker.stdout.readline, ''):
                    if line != '':
                        sys.stdout.write(line)
                    else:
                        break
            else:
                print('Worker %s completed!' % idx)
                for line in iter(worker.stdout.readline, ''):
                    if line != '':
                        sys.stdout.write(line)
                    else:
                        break
                workers[idx] = None

        is_all_completed = True
        for idx, worker in enumerate(workers):
            if worker is not None:
                is_all_completed = False
                break

        if is_all_completed:
            break
        time.sleep(0.1)


class PerfServerCMD(CLICommand):
    name = 'server'

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        """ define args for create pipeline template command.
        """
        parser = parsers.add_parser(PerfServerCMD.name)
        add_perf_args(parser)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        # start monitor
        p_monitor = start_monitor(self.args)
        # start tensorboard
        p_tensorboard = start_tensorboard(self.args)
        # start server
        p_server = start_server(self.args)

        wait_for_workers([p_monitor, p_tensorboard, p_server])
