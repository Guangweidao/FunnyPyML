# -*- coding: utf-8 -*-
import time


class Timer(object):
    def __init__(self):
        self._tic = time.time()
        self._tac = 0.
        self._duration = 0.
        self._time_format = '%Y-%m-%d %X'

    def stop(self):
        self._tac = time.time()
        self._duration = self._tac - self._tic

    def get_start_time(self):
        return time.strftime(self._time_format, time.localtime(self._tic))

    def get_end_time(self):
        return time.strftime(self._time_format, time.localtime(self._tac))

    def get_duration_time(self):
        return self._duration


class TimeScheduler(object):
    def __init__(self):
        self._task = dict()
        self._group = dict()
        self.create_group('default')

    def __setitem__(self, key, value):
        assert isinstance(value, Timer), 'value must be a Timer.'
        self._task.__setitem__(key, value)

    def __getitem__(self, item):
        return self._task.__getitem__(item)

    def create_group(self, group_name):
        self._group[group_name] = list()

    def start_task(self, task_name, group_name='default'):
        assert self._group.has_key(group_name), 'there is not a group:' + group_name
        self._task[task_name] = Timer()
        if task_name not in self._group[group_name]:
            self._group[group_name].append(task_name)

    def stop_task(self, task_name):
        assert task_name in self._task.keys(), 'task manager does not have ' + task_name
        self._task[task_name].stop()

    def tic_tac(self, task_name, func, **params):
        self.start_task(task_name=task_name)
        result = func(**params)
        self.stop_task(task_name=task_name)
        return result

    def get_group_time(self, group_name):
        return reduce(lambda x, y: x + y, [self._task[v].get_duration_time() for v in self._group[group_name]])

    def get_all_time(self):
        return reduce(lambda x, y: x + y, [v.get_duration_time() for v in self._task.values()])

    def print_task_schedule(self, task_name):
        print 'task %s:  From %s\tto %s\tduration %f s' % (
            task_name, self._task[task_name].get_start_time(), self._task[task_name].get_end_time(),
            self._task[task_name].get_duration_time())

    def print_group_schedule(self, group_name):
        assert group_name in self._group, 'no group: ' + group_name
        if len(self._group[group_name]) > 0:
            for task_name in self._group[group_name]:
                print '%s.%s:  From %s\tto %s\tduration %f s' % (group_name,
                                                                 task_name, self._task[task_name].get_start_time(),
                                                                 self._task[task_name].get_end_time(),
                                                                 self._task[task_name].get_duration_time())

    def print_all_schedule(self):
        for group_name in self._group.keys():
            self.print_group_schedule(group_name)

    def timing(self, task_name, group_name='default'):
        def _timing(func):
            def wrapper(*args, **kwargs):
                self.start_task(task_name, group_name)
                result = func(*args, **kwargs)
                self.stop_task(task_name)
                return result

            return wrapper

        return _timing
