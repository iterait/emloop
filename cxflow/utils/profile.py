import timeit
import typing


class Timer(object):
    TimeProfile = typing.NewType('TimeProfile', typing.Dict[str, typing.List[float]])

    def __init__(self, name: str, profile: TimeProfile):
        self._name = name
        self._profile = profile

    def __enter__(self):
        self._start = timeit.default_timer()

    def __exit__(self, *args):
        end = timeit.default_timer()
        if self._name not in self._profile:
            self._profile[self._name] = []
        self._profile[self._name].append(end - self._start)
