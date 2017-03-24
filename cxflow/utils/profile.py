import timeit
import typing


class Timer(object):
    TimeProfile = typing.NewType('TimeProfile', typing.Dict[str, typing.List[float]])

    def __init__(self, name: str, profile: TimeProfile):
        self.name = name
        self.profile = profile

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, *args):
        end = timeit.default_timer()
        if self.name not in self.profile:
            self.profile[self.name] = []
        self.profile[self.name].append(end - self.start)
