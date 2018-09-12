import os
import pickle
import datetime
import pytz
import json

import yaml
import numpy
import pandas


def load_data(pathname):
    if pathname.endswith(".pickle"):
        with open(pathname, "rb") as f:
            d = pickle.load(f)
    elif pathname.endswith(".csv"):
        d = pandas.read_csv(pathname)
    elif pathname.endswith(".json"):
        with open(pathname, 'r') as f:
            d = json.load(f)
    elif pathname.endswith(".txt"):
        with open(pathname, 'r') as f:
            d = f.read()
    else:
        extension = os.path.basename(pathname).split(".")[-1]
        raise Exception('Unrecognized file extension: "{}"'.format(extension))
    return d


def save_data(data_object, pathname, protocol=2, index=False):
    if pathname.endswith(".pickle"):
        with open(pathname, "wb") as f:
            pickle.dump(data_object, f, protocol=protocol)
    elif pathname.endswith(".csv") and isinstance(data_object, pandas.DataFrame):
        data_object.to_csv(pathname, index=index)
    elif pathname.endswith(".json") and (isinstance(data_object, dict) or (
        isinstance(data_object, list) and numpy.all([isinstance(x, dict) for x in data_object]))):
        with open(pathname, 'w') as f:
            f.write(json.dumps(data_object))
    elif pathname.endswith(".txt") and isinstance(data_object, str):
        with open(pathname, 'w') as f:
            f.write(data_object)
    else:
        extension = os.path.basename(pathname).split(".")[-1]
        object_type = type(data_object)
        raise Exception('Unrecognized file extension "{}" and type "{}"'.format(extension, object_type))

def utc_now(days=0, hours=0, minutes=0, seconds=0):
    """Returns a timestamp of the current UTC time with optional offset (if specified)."""
    delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return (datetime.datetime.now(pytz.utc) + delta).strftime('%Y-%m-%dT%H:%M:%SZ')


def time_to_str(time_obj):
    return time_obj.strftime('%Y-%m-%dT%H:%M:%SZ')

def load_yaml_config(filename):
    with open(filename, 'r') as file:
        return yaml.load(file)

def create_folder(fileName):
    """Utility function to create folder for output"""
    absolute_dirname = os.path.dirname(fileName)
    last_element = os.path.basename(absolute_dirname)
    if '.' in last_element: # not a folder
        absolute_dirname = os.path.dirname(absolute_dirname)
    if not os.path.exists(absolute_dirname):
        os.makedirs(absolute_dirname)
