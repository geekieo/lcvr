# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
from tensorflow.python.client import device_lib

import train
import logger

def test_local_device_protos():
  logging = logger.Logging('test_train','test_lcvr')
  logging.info('测试')

  # test 
  local_device_protos = device_lib.list_local_devices()
  logging.debug(local_device_protos)
