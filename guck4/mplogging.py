import logging
import logging.handlers
import multiprocessing
import sys
import traceback
import inspect
from setproctitle import setproctitle
import os
import signal
from guck4 import __appabbr__


class SigHandler_ll:
    def __init__(self):
        pass

    def sighandler_ll(self, a, b):
        pass


def whoami():
    outer_func_name = str(inspect.getouterframes(inspect.currentframe())[1].function)
    outer_func_linenr = str(inspect.currentframe().f_back.f_lineno)
    return outer_func_name + " / #" + outer_func_linenr + ": "


def logging_listener(queue, level, filename):
    setproctitle(__appabbr__ + ".loglistener.py")
    sh = SigHandler_ll()
    signal.signal(signal.SIGINT, sh.sighandler_ll)
    signal.signal(signal.SIGTERM, sh.sighandler_ll)
    # root = logging.getLogger()
    h = logging.FileHandler(filename, mode="w")
    # f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f = logging.Formatter("%(asctime)s - %(levelname)s / %(filename)s:%(lineno)s (%(funcName)s) - '%(message)s'",
                          "%d-%m-%Y %H:%M:%S")
    h.setFormatter(f)
    # root.addHandler(h)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            if record.levelno < level:
                continue
            logger = logging.getLogger(record.name)
            logger.addHandler(h)
            logger.propagate = False
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def logtest(logger):
    logger.debug("Test1")
    logger.info("test2")


def setup_logger(queue, modname):
    logger = logging.getLogger(os.path.basename(modname).split(".py")[0])
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    fh = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    logger.addHandler(fh)
    return logger


def start_logging_listener(filename, maxlevel=logging.DEBUG):
    mp_loggerqueue = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(target=logging_listener, args=(mp_loggerqueue, maxlevel, filename, ))
    listener.start()
    return mp_loggerqueue, listener


def stop_logging_listener(queue, listener):
    queue.put(None)
    listener.join(timeout=2)


if __name__ == '__main__':
    setproctitle(__appabbr__ + "." + os.path.basename(__file__))

    # starts loglistener
    filename = "mptest.log"
    mp_loggerqueue, loglistener = start_logging_listener(filename, maxlevel=logging.DEBUG)

    # setups local logger which logs to loglistener
    logger = setup_logger(mp_loggerqueue, __file__)
    logtest(logger)
    logger.warning("Attention")

    # stop loglistener
    stop_logging_listener(mp_loggerqueue, loglistener)
