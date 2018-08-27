# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:41:32 2017

@author: akihito
"""


def error_message(func_name, varname, value, msg):
    return '{}: {} is {}, {}'.format(func_name, varname, value, msg)
