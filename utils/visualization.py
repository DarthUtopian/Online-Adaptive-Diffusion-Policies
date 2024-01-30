import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt  
import gym
from typing import Dict, List, Tuple, Union, Optional
import utils

def plot_results(metric: Dict[str, np.ndarray], 
                x_label: str, 
                y_label: str, 
                title: str, 
                save_name: str,
                save_dir: str, 
                legend: List[str] = None, 
                ylim: Tuple[float, float] = None,
                smooth: int = 1,
                show: bool = False) -> None:
    
    plt.figure()
    for key in metric.keys():
        if smooth > 1:
            metric[key] = np.convolve(metric[key], np.ones(smooth), 'valid') / smooth
        plt.plot(metric[key], label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend is not None:
        plt.legend(legend)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.savefig('{}/{}.png'.format(save_dir, save_name))
    
    if show:
        plt.show()
    plt.close()