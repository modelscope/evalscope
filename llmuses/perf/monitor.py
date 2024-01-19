from typing import Optional, Union, Tuple, Dict
import time
import argparse

import psutil
from pynvml import *

import torch
from torch.utils.tensorboard import SummaryWriter

class VisualizationBase:
    def write_scalar(self, group_name, metrics, step, walltime):
        raise NotImplemented


class TensorboardVisualization:
    def __init__(self, log_path: str):
        self._log_path = log_path
        self.writer = SummaryWriter(log_path)
    
    def write_scalar(self, group_name, metrics, step, walltime):
        for metric in metrics:
            self.writer.add_scalar('%s/%s'%(group_name, metric), metrics[metric], global_step=step, walltime=walltime)


def log_cpu_percent():
    cpu_percent = psutil.cpu_times_percent(1)
    current_time = time.time()
    metrics = [name for name in dir(cpu_percent) if not name.startswith('__') and not name.startswith('_')]
    metrics_value = {}
    for metric in metrics:
        if not callable(getattr(cpu_percent, metric)):
            metrics_value[metric] = getattr(cpu_percent, metric)
    total_percent = psutil.cpu_percent(interval=1)
    metrics_value['cpu_percent'] = total_percent
    return metrics_value

class PyNVMLContext:
    def __enter__(self):
        nvmlInit()

    def __exit__(self, type, value, traceback):
        nvmlShutdown()


def device_info(gpus=None) -> Dict:
    if gpus is None:
        gpus = list(range(0, torch.cuda.device_count()))
    else:
        if len(args.gpus.split(",")):
            gpus = args.gpus.split(",")
        else:
            gpus = [torch.cuda.current_device()]
    gpu_metrics = {}
    for index in gpus:
        with PyNVMLContext():
            handle = nvmlDeviceGetHandleByIndex(index)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_utilization = nvmlDeviceGetUtilizationRates(handle)
        gpu_metrics['%s/gpu_percent'%index] = gpu_utilization.gpu
        gpu_metrics['%s/memory_percent'%index] = gpu_utilization.memory
        gpu_metrics['%s/memory_used'%index] = mem_info.used
        gpu_metrics['%s/memory_free'%index] = mem_info.free
        gpu_metrics['%s/memory_total'%index] = mem_info.total

    return gpu_metrics

visualization_backends: Dict[str, type[VisualizationBase]] = {
    'tensorboard': TensorboardVisualization
}

def main(args):
    visualizations = {}
    for vis in args.visualization:
        if vis in visualization_backends:
            visualizations[vis] = visualization_backends[vis](args.logdir)
        else:
            raise Exception("Unsupport visualization %s"%vis)
    step = 0
    while True:
        current_time = time.time()
        cpu_metrics = log_cpu_percent()
        print(cpu_metrics, flush=True)
        gpu_metrics = device_info(gpus=args.gpus)
        print(gpu_metrics, flush=True)
        for vis in args.visualization:
            visualizations[vis].write_scalar('cpu', cpu_metrics, step, current_time)
            visualizations[vis].write_scalar('gpu', gpu_metrics, step, current_time)
        time.sleep(0.99)
        step += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor system metrics.")
    parser.add_argument("--logdir", type=str, required=True,
                        help="Where the metrics are logged")
        
    parser.add_argument('--visualization', type=str, nargs='+', default=['tensorboard'],
                        help='Visualization system, default tensorboard, only support tensorboard currently')
    parser.add_argument("--gpus", type=str, default=None,
                        help="A single GPU like 1 or multiple GPUs like 0,2",)
    args = parser.parse_args()
    main(args)                    

