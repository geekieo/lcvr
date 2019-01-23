# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
from tensorflow.python.client import device_lib

import train
import logger

logging = logger.Logging('test_train','test_lcvr')
logging.info('测试')

# test 
local_device_protos = device_lib.list_local_devices()
logging.debug(local_device_protos)
'''
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 3247944551154579203
]
'''