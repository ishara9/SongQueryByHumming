import time

log_timer = 0

def log_time(event=""):
    global localtime
    localtime = time.localtime(time.time())
    print('log INFO %s time:  %d-%d-%d %d:%d:%d' % (event,
                                                    localtime.tm_mday, localtime.tm_mon, localtime.tm_year,
                                                    localtime.tm_hour, localtime.tm_min, localtime.tm_sec))
