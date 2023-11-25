import cv2
from setproctitle import setproctitle
from guck4 import mplogging, __appabbr__
import os
import time
import signal
import numpy as np
import sys
import inspect
from threading import Thread
import threading
import time
import queue

CNAME = None
TERMINATED = False


def whoami():
    outer_func_name = str(inspect.getouterframes(inspect.currentframe())[1].function)
    outer_func_linenr = str(inspect.currentframe().f_back.f_lineno)
    return outer_func_name + " " + str(CNAME) + " / #" + outer_func_linenr + ": "


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


class SigHandler_mpcam:
    def __init__(self, logger):
        self.logger = logger

    def sighandler_mpcam(self, a, b):
        global TERMINATED
        TERMINATED = True


class Detection:
    def __init__(self, id, frame, t, rect, descr, cd, ca):
        self.id = id
        self.rect = rect
        self.class_detection = cd
        self.class_detection_lt = t
        self.class_ai = ca
        self.class_ai_lt = 0
        self.frame = frame
        self.t = t
        self.descrkp = None
        self.descrdes = None
        self.descriptor = descr
        self.calcHog_descr()

    def calcHog_descr(self):
        x, y, w, h = self.rect
        self.descrkp, self.descrdes = self.descriptor.detectAndCompute(self.frame[y:y + h, x:x + w], None)
        return


class NewMatcherThread(Thread):
    def __init__(self, cfg, options, logger):
        Thread.__init__(self)
        self.lock = threading.Lock()
        self.options = options
        self.logger = logger
        self.SURL = cfg["stream_url"]
        self.NAME = cfg["name"]
        self.YMAX0 = self.XMAX0 = None
        self.MINAREA = cfg["min_area_rect"]
        self.CAP = None
        self.MOG2SENS = cfg["mog2_sensitivity"]
        self.HIST = 800 + (5 - self.MOG2SENS) * 199
        self.KERNEL2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))
        self.NIGHTMODE = False
        self.running = False
        self.startup = True
        self.ret = False
        self.frame = None
        self.queue = queue.Queue()
        self.frame_grabbed = threading.Event()
        self.NO_GPUS = 0
        # get target_fps
        try:
            self.target_fps = int(self.options["target_fps"])
            if self.target_fps < 1:
                self.target_fps = 1
        except Exception as e:
            self.target_fps = 8
        self.logger.debug(whoami() + "Setting target_fps to " + str(self.target_fps))
        # get bgsubtractor
        try:
            self.bgsubtractor = self.options["bgsubtractor"]
            if self.bgsubtractor not in ["MOG2", "KNN", "CNT"]:
                self.bgsubtractor = "KNN"
        except:
            self.bgsubtractor = "KNN"
        self.logger.debug(whoami() + "Setting background subtractor to " + self.bgsubtractor)
        if self.bgsubtractor == "KNN":
            self.setFGBG_KNN()
        elif self.bgsubtractor == "CNT":
            self.setFGBG_CNT()
        elif self.bgsubtractor == "MOG2":
            self.setFGBG_MOG2()

    def OpenVideoCapture(self):
        ret = False
        self.startup = True
        self.CAP = None
        for i in range(10):
            if not self.CAP:
                try:
                    self.CAP = cv2.VideoCapture(self.SURL, cv2.CAP_FFMPEG)
                    ret, frame = self.CAP.read()
                    self.YMAX0, self.XMAX0 = frame.shape[:2]
                except:
                    pass
            if self.CAP.isOpened():
                ret = True
                break
            time.sleep(0.1)
        self.startup = False
        return ret

    def run(self):
        ret = self.OpenVideoCapture()
        self.running = True
        if not ret:
            self.CAP = None
            self.running = False
        while self.running:
            if not self.startup:
                try:
                    with self.lock:
                        ret = self.CAP.grab()
                        if ret:
                            self.frame_grabbed.set()
                except Exception as e:
                    self.logger.error(whoami() + "Cannot grab frame for " + self.NAME + ": " + str(e))
                    ret = False
                if not ret:
                    self.running = False
                time.sleep(0.01)

    def stop(self):
        self.running = False
        self.startup = True
        with self.lock:
            if self.CAP:
                self.CAP.release()

    def setFGBG_KNN(self):
        ms = self.MOG2SENS
        if not self.NIGHTMODE:
            hist = int(800 + (5 - ms) * 100)
        else:
            hist = int(500 + (5 - ms) * 70)
        self.FGBG = cv2.createBackgroundSubtractorKNN(history=hist, detectShadows=False)
        self.logger.info(whoami() + "Created BackgroundSubtractorKNN")
        return

    def setFGBG_CNT(self):
        self.FGBG = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=self.target_fps, useHistory=True,
                                                             maxPixelStability=self.target_fps * 60,
                                                             detectShadows=False)
        self.logger.info(whoami() + "Created BackgroundSubtractorCNT")
        return

    # MOG2 is on GPU if possible
    def setFGBG_MOG2(self):
        try:
            self.NO_GPUS = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            self.NO_GPUS = 0
        # no GPU -> MOG2 on CPU
        if self.NO_GPUS == 0:
            self.setMOG2_ON_CPU()
        # on GPU (with CPU as fallback)
        else:
            try:
                self.FGBG_GPU = cv2.cuda.createBackgroundSubtractorMOG2()
                self.CUDA_STREAM_0 = cv2.cuda_Stream()
                self.FRAME_CUDA = None
                self.logger.info(whoami() + "Created BackgroundSubtractorMOG2 on GPU")
            except Exception as e:
                self.NO_GPUS = 0
                self.setMOG2_ON_CPU()
                self.logger.warning(whoami() + "Created BackgroundSubtractorMOG2 on CPU, GPU setup failed!")
        return

    def setMOG2_ON_CPU(self):
        ms = self.MOG2SENS
        if not self.NIGHTMODE:
            hist = int(800 + (5 - ms) * 100)
        else:
            hist = int(500 + (5 - ms) * 70)
        self.FGBG = cv2.createBackgroundSubtractorMOG2(history=hist, detectShadows=False)
        self.logger.info(whoami() + "Created BackgroundSubtractorMOG2 on CPU")

    def get_caption_and_process(self):
        ret = self.frame_grabbed.wait(2)
        if ret:
            with self.lock:
                ret, frame = self.CAP.retrieve()
                self.frame_grabbed.clear()
        if not ret:
            return False, None, None, time.time()
        # if MOG2/GPU
        if self.NO_GPUS > 0:
            try:
                if not self.FRAME_CUDA:  # ifFRAME_CUDA not init, do it now
                    self.FRAME_CUDA = cv2.cuda_GpuMat()
                    # self.median_cuda = cv2.cuda.createMedianFilter(cv2.CV_8U, 3);
                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))
                    # self.morphology_ex_cuda = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, self.FRAME_CUDA.type(), kernel)
                self.FRAME_CUDA.upload(frame)
                self.FRAME_CUDA = cv2.cuda.cvtColor(self.FRAME_CUDA, cv2.COLOR_RGB2BGR)
                self.FRAME_CUDA = self.FGBG_GPU.apply(self.FRAME_CUDA, 0.05, self.CUDA_STREAM_0)
                self.CUDA_STREAM_0.waitForCompletion()
                # self.median_cuda.apply(self.FRAME_CUDA, self.FRAME_CUDA, self.CUDA_STREAM_0)
                # self.CUDA_STREAM_0.waitForCompletion()
                fggray = self.FRAME_CUDA.download()
            except Exception as e:
                self.logger.warning(whoami() + str(e))
                return False, None, None, time.time()
        # else if on CPU
        else:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fggray = self.FGBG.apply(gray, 1 / self.HIST)
            except Exception as e:
                self.logger.warning(whoami() + str(e))
                return False, None, None, time.time()

        # all the other stuff besides bgsub is on cpu!!
        try:
            fggray = cv2.medianBlur(fggray, 5)
            edged = auto_canny(fggray)
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, self.KERNEL2)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts0 = [cv2.boundingRect(c) for c in cnts]
            rects = [(x, y, w, h) for x, y, w, h in cnts0 if w * h > self.MINAREA]
            return ret, rects, frame, time.time()
        except Exception as e:
            self.logger.warning(whoami() + str(e))
            return False, None, None, time.time()


def run_cam(cfg, options, child_pipe, mp_loggerqueue):
    global CNAME
    global TERMINATED

    cv2.setNumThreads(1)

    CNAME = cfg["name"]

    setproctitle(__appabbr__ + "." + cfg["name"] + "_" + os.path.basename(__file__))

    logger = mplogging.setup_logger(mp_loggerqueue, __file__)
    logger.info(whoami() + "starting ...")

    event_stopped = threading.Event()
    sh = SigHandler_mpcam(logger)
    signal.signal(signal.SIGINT, sh.sighandler_mpcam)
    signal.signal(signal.SIGTERM, sh.sighandler_mpcam)

    tm = NewMatcherThread(cfg, options, logger)
    tm.start()
    while tm.startup:
        time.sleep(0.03)
    child_pipe.recv()
    child_pipe.send((tm.running, tm.YMAX0, tm.XMAX0))
    if not tm.running:
        logger.error(whoami() + "cam is not working, aborting ...")
        sys.exit()

    waitnext = 0.01
    waitnext_delta = 0.01
    oldt = time.time()
    MAXFPS = tm.target_fps

    while not TERMINATED:
        try:
            cmd = child_pipe.recv()
            if cmd == "stop":
                child_pipe.send(("stopped!", None))
                break
            if cmd == "query":
                ret, rects, frame, t0 = tm.get_caption_and_process()
                if ret:
                    exp0 = (ret, frame, rects, t0)
                    child_pipe.send(exp0)
                    fps = 1 / (t0 - oldt)
                    if fps > MAXFPS:
                        waitnext += waitnext_delta
                    elif fps < MAXFPS and waitnext > waitnext_delta:
                        waitnext -= waitnext_delta
                    oldt = t0
                    time.sleep(waitnext)
                else:
                    logger.error(whoami() + "Couldn't capture frame in main loop!")
                    exp0 = (ret, None, [], None)
                    child_pipe.send(exp0)
                    continue
        except Exception as e:
            logger.error(whoami() + str(e))
            exp0 = (False, None, [], None)
            child_pipe.send(exp0)

    tm.stop()
    tm.join()

    logger.info(whoami() + tm.NAME + " exited!")
