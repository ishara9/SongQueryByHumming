import time


def log_time():
    global localtime
    localtime = time.localtime(time.time())
    print('log INFO time: %d-%d-%d %d:%d:%d' % (
        localtime.tm_mday, localtime.tm_mon, localtime.tm_year, localtime.tm_hour, localtime.tm_min, localtime.tm_sec))
