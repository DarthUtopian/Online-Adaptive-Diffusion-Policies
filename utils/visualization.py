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
                xlim: Tuple[float, float] = None,
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
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.savefig('{}/{}.png'.format(save_dir, save_name))
    
    if show:
        plt.show()
    plt.close()
    
    
def plot_training_curves(data: Dict[str, np.ndarray],
                         fig_name: str,
                         save_dir: str, 
                         iterations: np.ndarray = None,
                         fig_size = (5,5),
                         xlim = None,
                         ylim = None,
                         xlabel = None,
                         ylabel = None,
                         show: bool = False) -> None:
    fig = plt.figure(figsize=fig_size)
    for runs in data.keys():
        curr_data = data[runs]
        data_mean = np.mean(curr_data, axis=0)
        data_std = np.std(curr_data, axis=0)
        data_min = data_mean - 0.95 * data_std
        data_max = data_mean + 0.95 * data_std
        if iterations is not None:
            plt.plot(iterations, data_mean, label=runs)
            plt.fill_between(iterations, data_min, data_max, alpha=0.6)
        else:
            plt.plot(data_mean, label=runs)
            plt.fill_between(np.arange(data_mean.shape[0]), data_min, data_max, alpha=0.6)
        
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    #plt.legend()
    #plt.grid(True)
    plt.title(f'{fig_name}')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{fig_name}_curves.png')
    if show:
        plt.show()
    else:
        plt.close()
        