# Copyright (c) Alibaba, Inc. and its affiliates.

import subprocess

if __name__ == '__main__':
    cmd = f'TEST_LEVEL_LIST=0,1 python3 -m unittest discover tests'
    run_res = subprocess.run(cmd, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if run_res.returncode == 0:
        print(f'>>test_run_all stdout: {run_res.stdout}')
    else:
        print(f'>>test_run_all stderr: {run_res.stderr}')
