import os
import time

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

class Log(object):
    def __init__(self, log_dict) -> None:
        super().__init__()
        self.logDir = log_dict['logDir']
        if log_dict['clear']:
            self.clear()
        self.logfile = f'{self.logDir}/log_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.txt'
        f = open(self.logfile,'w')
        f.close()

    def clear(self):
        del_file(self.logDir)

    def log(self, str, out=True):
        """ Log a string in a file """
        with open(self.logfile,'a') as f:
            f.write(str+'\n')
        if out:
            print(str)