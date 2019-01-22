# -*- coding:utf-8 -*-
import os
from env_utils import get_server_info
from logger import Logging

logger = Logging(name='config', filename='lcvr').logger

production_ip = [
  '10.80.75.152', #GTX1080ti
  '10.80.76.152', #GTX1080ti
  '10.80.78.152', #GTX1080ti
]
development_ip = [
  '10.90.72.177', #GTX1080
  '10.80.97.152'] #GTX1080ti

server_info = get_server_info()

if server_info['ip'] in production_ip or server_info['ip'] in development_ip:
  # have GPU
  import tensorflow as tf

  # deepgaze 模型设置
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ignore memory warning
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  tfconfig = tf.ConfigProto()
  tfconfig.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
  # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.15   # GPU 显存占用率

  if server_info['ip'] in production_ip:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU 0 or 1
    logger.info('Production config loaded.')

  elif server_info['ip'] in development_ip:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # 指定 GPU 0 or 1
    logger.info('Development config loaded.')
else:
  # doesn't have GPU
  logger.info('No GPU config loaded.')


if __name__ == '__main__':
  try:
    logger.info(tfconfig.gpu_options.per_process_gpu_memory_fraction)
  except Exception as e:
    logger.info(e)
  logger.info(cache_size)

