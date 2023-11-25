from setproctitle import setproctitle
import os, datetime
from guck4 import __appabbr__, __version__
from .__main__ import mainloop
import datetime


def run():
    setproctitle(__appabbr__ + os.path.basename(__file__))
    print(str(datetime.datetime.now()) + "- beginning start-up procedure for GUCK " + str(__version__) + ", please wait ...")
    exitcode = 3
    trstr = "N/A"
    while exitcode == 3:
        exitcode = mainloop()
        if exitcode == 3:
            trstr = str(datetime.datetime.now()) + ": RESTART - "
        else:
            trstr = str(datetime.datetime.now()) + ": SHUTDOWN - "
        print(trstr + "GUCK exited with return code:", exitcode)
        if exitcode == 3:
            print(trstr + "Restarting GUCK ...")
            print()
    print(trstr + "Exit GUCK")