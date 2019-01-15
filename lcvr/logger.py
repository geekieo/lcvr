# -*- coding: utf-8 -*-
import datetime
import logging
import os
import time


class Logging:
    def __init__(self, name='main', filename='croplog', slevel=logging.INFO, flevel=logging.DEBUG, ):
        '''logger for this server
        Arg:
            name: str; name of logging in different module 
            filename: str; profix of log name
            slevel: int; log level for output to terminal
            flevel: int; log level for output to log file
        '''
        log_dir = './logs'
        # Stitching address
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        # log_dir 为项目内日志专属文件夹，不要误删其他文件夹
        self._clean_file(log_dir, keepdays=7)
        self.filename = '%s.%s.log' % (
            filename, datetime.datetime.now().strftime('%Y%m%d'))
        self.filepath = os.path.join(log_dir, str(self.filename))
        # 创建一个logger
        self.logger = logging.getLogger(name)
        # 防止同名 logger 重复写入
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s %(filename)s[%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
            # 创建终端日志处理器
            self.sh = logging.StreamHandler()
            self.sh.setFormatter(formatter)
            self.sh.setLevel(slevel)
            # 创建文件日志处理器
            self.fh = logging.FileHandler(
                filename=self.filepath, encoding='utf-8')
            self.fh.setFormatter(formatter)
            self.fh.setLevel(flevel)
            # 给logger添加handler
            self.logger.addHandler(self.sh)
            self.logger.addHandler(self.fh)

    @staticmethod
    def _clean_file(log_dir, keepdays=7):
        for parent, dirnames, filenames in os.walk(log_dir):
            for filename in filenames:
                fullname = parent + "/" + filename  # 文件全称
                createTime = int(os.path.getctime(fullname))  # 文件创建时间
                nDayAgo = (datetime.datetime.now() -
                           datetime.timedelta(days=keepdays))  # 当前时间的n天前的时间
                timeStamp = int(time.mktime(nDayAgo.timetuple()))
                if createTime < timeStamp:  # 创建时间在n天前的文件删除
                    os.remove(os.path.join(parent, filename))


if __name__ == '__main__':
    logger = Logging(name='logger', filename='croplog').logger
    logger.debug('一个debug信息')
    logger.info('一个info信息'+'; '+'两个info信息'+'; '+'三个info信息')
    logger.warn('一个warning信息')
    logger.error('一个error信息')
    logger.critical('一个致命critical信息')
