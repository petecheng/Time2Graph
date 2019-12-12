# -*- coding: utf-8 -*-
import time
import datetime


def convert_str2float(strr):
    if strr == '':
        return -1.0
    else:
        return float(strr)


def convert_str2int(strr):
    if strr == '':
        return -1
    else:
        return int(strr)


def get_month_in_year(timestamp):
    return int(time.localtime(timestamp).tm_mon) - 1


def get_day_in_month(timestamp):
    return int(time.localtime(timestamp).tm_mday) - 1


def get_day_in_year(timestamp):
    return int(time.localtime(timestamp).tm_yday) - 1


def get_year(timestamp):
    return int(time.localtime(timestamp).tm_year)


def format_time_from_str(time_str, tfmt):
    return int(time.mktime(time.strptime(time_str, tfmt)))


def generate_time_series_time(begin, end, tfmt, duration):
    ret = []
    d_begin = datetime.datetime.fromtimestamp(format_time_from_str(time_str=begin, tfmt=tfmt))
    d_end = datetime.datetime.fromtimestamp(format_time_from_str(time_str=end, tfmt=tfmt))
    while d_begin <= d_end:
        ret.append(d_begin.strftime(tfmt))
        if duration == 'day':
            d_begin += datetime.timedelta(days=1)
        elif duration == 'hour':
            d_begin += datetime.timedelta(hours=1)
        else:
            raise NotImplementedError()
    return ret


def generate_time_index(begin, end, tfmt, duration):
    ret = {}
    d_begin = datetime.datetime.fromtimestamp(format_time_from_str(time_str=begin, tfmt=tfmt))
    d_end = datetime.datetime.fromtimestamp(format_time_from_str(time_str=end, tfmt=tfmt))
    cnt = 0
    while d_begin <= d_end:
        ret[d_begin.strftime(tfmt)] = cnt
        cnt += 1
        if duration == 'day':
            d_begin += datetime.timedelta(days=1)
        elif duration == 'hour':
            d_begin += datetime.timedelta(hours=1)
        else:
            raise NotImplementedError()
    return ret


def transform_np2tsv(x, y, fpath):
    output = open(fpath, 'w')
    for k in range(len(y)):
        data = x[k]
        output.write('{}'.format(y[k]))
        for i in range(len(data)):
            for j in range(len(data[i])):
                output.write('\t{}'.format(data[i, j]))
        output.write('\n')
