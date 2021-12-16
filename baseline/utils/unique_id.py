class __Imports:
    import os
    import datetime

def timepid() -> str:
    return f'{__Imports.datetime.datetime.now().strftime("%Y-%m%d-%H%M")}-{__Imports.os.getpid()}'

