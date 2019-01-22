# -*- coding: utf-8 -*-
import uuid
import socket
import platform
import numpy as np

def get_server_info():
  '''
  return server_info: for example:
    {'ip': '172.30.160.53', 'servername': 'D3020-70YCG02', 'system': 'Windows', 'mac': 'f8:bc:12:58:b2:63', 'dist': '', 'CPU': {}}
      cpu is dict type, read form "/proc/cpuinfo" only in linux, each item contains: cpu[cpuid]=cpu_info
        cpu_info is dict type,which contains information about a physical core.
  '''

  server_info = {}
  mac_hexadecimal = uuid.UUID(int=uuid.getnode()).hex[-12:]
  server_info['mac'] = ":".join([mac_hexadecimal[e:e + 2]
                                 for e in np.arange(0, 11, 2)])  # server MAC
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    # server IP
    server_info['ip'] = s.getsockname()[0]
  finally:
    s.close()
  server_info['name'] = platform.node()  # servername
  # operating system: windows / linux / mac os
  server_info['system'] = platform.system()
  # distributions name: cent os / ubuntu ...
  server_info['dist'] = platform.dist()[0]

  # if OS is linux try to get CPU info by reading file
  cpu = {}
  if server_info['system'] == 'Linux':
    cpu_info = {}
    cpu_num = 0
    with open("/proc/cpuinfo") as f:
      lines = f.readlines()
      for line in lines:
        if line == '\n':  # Multiple CPU
          cpu['cpu' + str(cpu_num)] = cpu_info
          cpu_info = {}
          cpu_num += 1
        else:
          name = line.split(':')[0].rstrip()
          var = line.split(':')[1].strip()
          cpu_info[name] = var

  server_info['CPU'] = cpu
  return server_info


if __name__ == "__main__":
  server_info = get_server_info()
  print(server_info)
