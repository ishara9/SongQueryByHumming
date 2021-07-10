import time

log_timer = 0


def log_time(event=""):
    global localtime
    localtime = time.localtime(time.time())
    print('log INFO %d-%d-%d %d:%d:%d: %s' % (
        localtime.tm_mday, localtime.tm_mon, localtime.tm_year,
        localtime.tm_hour, localtime.tm_min, localtime.tm_sec, event))
