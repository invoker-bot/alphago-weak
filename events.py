from typing import NoReturn, Callable


class Event:
    def __init__(self):
        self.events = {}

    def emit(self, event: str, *args, **kwargs) -> NoReturn:
        if event in self.events:
            for func in self.events[event]:
                func(*args, **kwargs)
    
    def subscribe(self, event: str, func: Callable):
        if event not in self.events:
            self.events[event] = []
        self.events[event].append(func)
        return id(func)

