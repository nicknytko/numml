import time


class Profiler:
    '''
    A small profiler class for measuring how long a block of code runs.

    This should be used like:
    ```
    with Profiler('label'):
        # code to run here
    ```
    and will automatically print runtime information.  Additionally, if
    multiple `with` blocks are nested they will be displayed hierarchically.
    '''

    enabled = True
    current = None
    tab_width = 2

    def __init__(self, section_name):
        self.section_name = section_name
        self.children = []

    def __enter__(self):
        if not Profiler.enabled:
            return

        self.parent = Profiler.current
        if self.parent is not None:
            self.parent.children.append(self)
        Profiler.current = self

        self.start_time = time.time()
        return self

    def print_recursive(self, level):
        print(((level * Profiler.tab_width) * ' ') + f'[{self.section_name}] {self.running_time:.5f}s')
        for child in self.children:
            child.print_recursive(level + 1)

    def __exit__(self, type, value, tb):
        if type is not None:
            raise value

        if not Profiler.enabled:
            return True

        self.running_time = time.time() - self.start_time

        if self.parent is None:
            self.print_recursive(0)
        Profiler.current = self.parent

        return True
