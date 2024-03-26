class StopExecution(Exception):
    def _render_traceback_(self):
        return []


def exit(): raise StopExecution
