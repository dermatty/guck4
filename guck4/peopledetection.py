import os
import queue
from setproctitle import setproctitle
from .mplogging import whoami
from guck4 import mplogging, mpcam, clear_all_queues, ConfigReader, __appabbr__
import time
import cv2
import torch
import torchvision
from torchvision import transforms
import PIL
import multiprocessing as mp
import signal
import numpy as np
import sys
from datetime import datetime
from threading import Thread, Lock
import warnings


# todo:
#    each camera own thread which gets data from camera and does peopledetection


TERMINATED = False


class SigHandler_pd:
    def __init__(self, logger):
        self.logger = logger

    def sighandler_pd(self, a, b):
        self.shutdown()

    def shutdown(self):
        global TERMINATED
        TERMINATED = True
        self.logger.debug("got signal, exiting ...")


class TorchResNet:
    def __init__(self, dirs, cfgr, logger):
        self.logger = logger
        self.active = False
        self.cfgr = cfgr
        self.dirs = dirs
        self.ai_conf = self.cfgr.get_ai()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.RESNETMODEL = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=self.weights).to(self.device)
                # self.RESNETMODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
                self.RESNETMODEL.eval()
                self.active = True
                self.logger.info("Torchvision Resnet initialized!")
            except Exception as e:
                self.logger.error(str(e) + ": cannot init Torchvision Resnet!")
                self.RESNETMODEL = None

    def image_loader(self, img_cv2):
        img_cv20 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img_cv20)
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(img).to(self.device)
        return image

    def get_cnn_classification(self, camera):
        if (not self.active) or not (camera.active and camera.frame is not None) or \
                (len(camera.rects) == 0 or self.RESNETMODEL is None):
            return []
        try:
            t0 = time.time()
            # self.logger.debug("-------- >>> " + camera.cname + ": performing resnet classification with " + str(len(camera.rects)) + " opencv detections ...")

            # get min & max value from rects
            camera.cnn_classified_list = []
            YMAX0, XMAX0 = camera.frame.shape[:2]
            x0 = XMAX0
            y0 = YMAX0
            x1 = 0
            y1 = 0
            for x, y, w, h in camera.rects:
                if x < x0:
                    x0 = x
                if x + w > x1:
                    x1 = x + w
                if y < y0:
                    y0 = y
                if y + w > y1:
                    y1 = y + h
            x0 = max(int(x0 - XMAX0 * 0.1), 0)
            x01 = min(int(x1 + XMAX0 * 0.1), XMAX0)
            y0 = max(int(y0 - YMAX0 * 0.1), 0)
            y01 = min(int(y1 + YMAX0 * 0.1), YMAX0)

            # crop frame and run cnn
            frame = camera.frame.copy()
            frame = frame[y0:y01, x0:x01]
            img0 = self.image_loader(frame)
            pred = self.RESNETMODEL([img0])[0]
            boxes0 = pred["boxes"].to("cpu").tolist()
            labels0 = pred["labels"].to("cpu").tolist()
            scores0 = pred["scores"].to("cpu").tolist()

            # only keep a few categories
            #allowed_categories = ["person", "dog", "cat"]
            allowed_idx0 = [(i, label) for i, label in enumerate(labels0) if
                           self.weights.meta["categories"][label] in self.ai_conf]
            allowed_idx = [i for i, label in allowed_idx0 if scores0[i] >
                           self.ai_conf[self.weights.meta["categories"][label]]]
            if allowed_idx:
                s0 = [str(scores0[i]) for i in allowed_idx]
                self.logger.info("Camera " + camera.cname + ": detection with prob.:" + str(s0))
            else:
                return
            # perform nms (remove useless boxes)
            boxes1 = torch.tensor(
                [[boxes0[i][0] + x0, boxes0[i][1] + y0, boxes0[i][2] + x0, boxes0[i][3] + y0] for i in allowed_idx])
            labels1 = torch.tensor([labels0[i] for i in allowed_idx])
            scores1 = torch.tensor([scores0[i] for i in allowed_idx])
            kept_idx = torchvision.ops.batched_nms(boxes1, scores1, labels1, 0.5)

            boxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for i, box in enumerate(boxes1.tolist()) if
                     torch.tensor(i) in kept_idx]
            labels = [self.weights.meta["categories"][label] for i, label in enumerate(labels1.tolist()) if
                      torch.tensor(i) in kept_idx]

            camera.cnn_classified_list = [(box[0], box[1], box[2] - box[0], box[3] - box[1], labels[i], scores0[i])
                                          for i, box in enumerate(boxes)]
        except Exception as e:
            self.logger.error(str(e) + camera.cname + ": ResNet classification error!")
            camera.cnn_classified_list = []
        alarmstr = ""
        for _, _, _, _, label, score in camera.cnn_classified_list:
            alarmstr += camera.cname + ": " + label + " (" + str(round(score,3) * 100) + "%)"
        return alarmstr


class Camera(Thread):
    def __init__(self, ccfg, dirs, options, mp_loggerqueue, logger):
        Thread.__init__(self)
        self.daemon = True
        self.ccfg = ccfg
        self.options = options
        self.parent_pipe, self.child_pipe = mp.Pipe()
        self.mpp = None
        self.outvideo = None
        self.ymax = -1
        self.xmax = -1
        self.dirs = dirs
        self.is_recording = False
        self.recordfile = None
        self.frame = None
        self.oldframe = None
        self.rects = []
        self.tx = None
        self.shutdown_completed = False
        self.running = False
        self.newframe = False
        self.cnn_classified_list = []
        self.fpslist = []
        self.lock = Lock()
        self.startup_completed = False

        self.logger = logger
        self.mp_loggerqueue = mp_loggerqueue

        try:
            self.isok = True
            self.cname = ccfg["name"]
            self.active = ccfg["active"]
            self.stream_url = ccfg["stream_url"]
            self.photo_url = ccfg["photo_url"]
            self.reboot_url = ccfg["reboot_url"]
            self.ptz_mode = ccfg["ptz_mode"]
            self.ptz_right_url = ccfg["ptz_right_url"]
            self.ptz_left_url = ccfg["ptz_left_url"]
            self.ptz_up_url = ccfg["ptz_up_url"]
            self.ptz_down_url = ccfg["ptz_down_url"]
            self.min_area_rect = ccfg["min_area_rect"]
            self.hog_scale = ccfg["hog_scale"]
            self.hog_thresh = ccfg["hog_thresh"]
            self.mog2_sensitivity = ccfg["mog2_sensitivity"]
        except Exception as e:
            self.logger.error(str(e))

        try:
            self.fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        except Exception:
            self.logger.error("Cannot get fourcc, no recording possible")
            self.fourcc = None

    def get_fps(self):
        with self.lock:
            if len(self.fpslist) == 0:
                fps = 0
            else:
                fps = sum([f for f in self.fpslist]) / len(self.fpslist)
                if len(self.fpslist) > 20:
                    del self.fpslist[0]
        return fps

    def shutdown(self):
        self.logger.debug("camera " + self.cname + " starting shutdown initialize")
        if self.outvideo:
            self.outvideo.release()
            self.logger.debug("camera " + self.cname + " video released!")
        try:
            self.parent_pipe.send("stop")
            t0 = time.time()
            polled = False
            while not polled and time.time() - t0 < 10:
                polled = self.parent_pipe.poll()
                time.sleep(0.05)
            if polled:
                ret, _ = self.parent_pipe.recv()
                self.mpp.join(5)
                if self.mpp.is_alive():
                    os.kill(self.mpp.pid, signal.SIGTERM)
            else:
                if self.mpp:
                    if self.mpp.is_alive():
                        os.kill(self.mpp.pid, signal.SIGTERM)
        except Exception:
            if self.mpp:
                if self.mpp.is_alive():
                    os.kill(self.mpp.pid, signal.SIGTERM)
        self.mpp = None
        self.logger.debug("camera " + self.cname + " mpp stopped!")
        try:
            cv2.destroyWindow(self.cname)
        except Exception:
            pass
        self.stop_recording()
        self.frame = None
        self.logger.debug("camera " + self.cname + " shutdown finished!")
        return 1

    def startup_cam(self):
        if not self.active or not self.isok:
            return None
        self.mpp = mp.Process(target=mpcam.run_cam,
                              args=(self.ccfg, self.options, self.child_pipe, self.mp_loggerqueue,))
        self.mpp.start()
        try:
            self.parent_pipe.send("query_cam_status")
            self.isok, self.ymax, self.xmax = self.parent_pipe.recv()
        except Exception:
            self.isok = False
            self.active = False
        if self.isok:
            self.logger.debug("camera " + self.cname + " started!")
        else:
            self.logger.debug("camera " + self.cname + " out of function, not started!")
            self.mpp.join()
            self.mpp = None
        return self.mpp

    def stop(self):
        if self.shutdown_completed and not self.mpp:
            self.logger.warning(self.cname + " shutdown already completed, exiting ...")
            return 1
        elif self.shutdown_completed and self.mpp:
            self.logger.warning(self.cname + " shutdown only half completed, aborting !!!")
            return -1
        self.logger.debug("setting stop for " + self.cname)
        self.running = False
        t0 = time.time()
        while not self.shutdown_completed and time.time() - t0 < 10:
            time.sleep(0.1)
        if not self.shutdown_completed:
            self.logger.error(self.cname + " shutdown sequence timed out, aborting !!!!")
        else:
            self.logger.debug("shutdown completed for " + self.cname)

    def run(self):
        if not self.active or not self.isok:
            self.startup_completed = True
            return
        self.startup_cam()
        self.startup_completed = True
        if not self.isok or not self.active or not self.mpp:
            return
        self.running = True
        while self.running and self.isok and self.active:
            t_query = time.time()
            try:
                self.parent_pipe.send("query")
            except Exception as e:
                self.logger.warning(str(e) + ": error in communication (pipe_send)  with camera " + self.cname)
                self.newframe = False
                time.sleep(1.0)
                continue
                #self.running = False
                #self.isok = False
                #break
            while True:
                cond1 = self.running
                cond2 = self.parent_pipe.poll()
                if not cond1 or cond2:
                    break
                time.sleep(0.01)
            if not cond1:
                break
            if not cond2:
                self.newframe = False
                time.sleep(1.0)
                continue
            #if cond1 and not cond2:  # stopped and no poll
            #    break
            ret, frame0, rects, tx = self.parent_pipe.recv()
            self.tx = tx
            t_reply = time.time()
            with self.lock:
                self.fpslist.append(1 / (t_reply - t_query))
            if not cond1:
                break
            if not ret:
                self.logger.warning(": error in communication (ret) with camera " + self.cname)
                self.newframe = False
                time.sleep(1.0)
                continue
                #self.running = False
                #break
            else:
                if self.frame is not None:
                    self.oldframe = self.frame.copy()
                else:
                    self.oldframe = None
                self.frame = frame0.copy()
            self.newframe = False
            if self.frame is not None and self.oldframe is not None:
                if np.bitwise_xor(self.frame, self.oldframe).any():
                    self.newframe = True
            self.rects = rects
        self.shutdown()
        self.shutdown_completed = True
        self.logger.debug(": camera " + self.cname + " - thread completed!")

    def get_new_detections(self, cnn=True):
        if cnn:
            return self.cnn_classified_list
        else:
            return self.rects

    def clear_new_detections(self):
        self.cnn_classified_list = []
        self.rect = []

    def draw_text(self, img, text,
                  font=cv2.FONT_HERSHEY_PLAIN,
                  pos=(0, 0),
                  font_scale=3.0,
                  font_thickness=1,
                  text_color=(255, 255, 255),
                  text_color_bg=(0, 0, 0)
                  ):
        x,y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color,
                    thickness=font_thickness)

        return text_size

    def draw_detections(self, cnn=True):
        if cnn:
            rects = self.cnn_classified_list
        else:
            rects = self.rects
        if self.frame is not None:
            ymax0, xmax0 = self.frame.shape[:2]
            # draw detections
            for x, y, w, h, category_name, score in rects:
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, xmax0)
                y2 = min(y + h, ymax0)
                outstr = category_name + " / " + str(round(score, 3) *100) + "%"
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                self.draw_text(self.frame, outstr, pos=(x1 + 4, y2 - 14), font_scale=1)
                #cv2.putText(self.frame, outstr, (x1 + 4, y2 - 14), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0),
                #            thickness=1)

    def start_recording(self):
        if not self.active or not self.isok:
            return None
        if not self.fourcc:
            self.is_recording = False
            self.logger.debug("camera " + self.cname + " no recording possible due to missing fourcc/codec!")
        if self.outvideo:
            try:
                self.outvideo.release()
            except Exception:
                pass
        now = datetime.now()
        datestr = now.strftime("%d%m%Y-%H:%M:%S")
        self.recordfile = self.dirs["video"] + self.cname + "_" + datestr + ".avi"
        self.outvideo = cv2.VideoWriter(self.recordfile, self.fourcc, 10.0, (self.xmax, self.ymax))
        self.is_recording = True
        self.logger.debug("camera " + self.cname + " recording started: " + self.recordfile)

    def write_record(self):
        if not self.active or not self.isok:
            return None
        if self.outvideo and self.is_recording:
            self.outvideo.write(self.frame)

    def stop_recording(self):
        if self.outvideo:
            self.outvideo.release()
            self.outvideo = None
        self.is_recording = False
        self.logger.debug("camera " + self.cname + " recording stopped")


def restart_cam(cameras, cname0):
    for c in cameras:
        if c.cname == cname0 and c.active():
            c.stop()
            c.start()
            while not c.startup_completed:
                time.sleep(0.05)
            if not c.mpp:
                c.join(timeout=3)


def shutdown_cams(cameras):
    for c in cameras:
        if c.active:
            c.stop()


def startup_cams(cameras):
    for c in cameras:
        c.start()
        while not c.startup_completed:
            time.sleep(0.05)
        if not c.mpp:
            c.join(timeout=3)


def stop_cams(cameras):
    for c in cameras:
        c.stop()


def destroy_all_cam_windows(cameras):
    for c in cameras:
        try:
            cv2.destroyWindow(c.cname)
        except Exception:
            continue


def start_all_recordings(cameras):
    for c in cameras:
        c.start_recording()


def stop_all_recordings(cameras):
    for c in cameras:
        c.stop_recording()


def run_cameras(pd_outqueue, pd_inqueue, dirs, cfg, mp_loggerqueue):
    global TERMINATED

    setproctitle(__appabbr__ + "." + os.path.basename(__file__))

    # tf.get_logger().setLevel('INFO')
    # tf.autograph.set_verbosity(1)

    logger = mplogging.setup_logger(mp_loggerqueue, __file__)
    logger.info("starting ...")

    sh = SigHandler_pd(logger)
    signal.signal(signal.SIGINT, sh.sighandler_pd)
    signal.signal(signal.SIGTERM, sh.sighandler_pd)

    cfgr = ConfigReader(cfg)
    camera_config = cfgr.get_cameras()
    options = cfgr.get_options()
    cameras = []
    for c in camera_config:
        camera = Camera(c, dirs, options, mp_loggerqueue, logger)
        cameras.append(camera)

    startup_cams(cameras)
    logger.info("all cameras started!")

    tgram_active = False
    kbd_active = False
    pd_in_cmd, pd_in_param = pd_inqueue.get()
    if pd_in_cmd == "tgram_active":
        tgram_active = pd_in_param
    elif pd_in_cmd == "kbd_active":
        kbd_active = pd_in_param
    if not camera_config or not cameras:
        logger.error("cannot get correct config for cameras, exiting ...")
        pd_outqueue.put(("error:config", None))
        sys.exit()
    else:
        pd_outqueue.put(("allok", None))

    torchresnet = TorchResNet(dirs, cfgr, logger)
    try:
        showframes = (options["showframes"].lower() == "yes")
    except Exception:
        logger.warning("showframes not set in config, setting to default False!")
        showframes = False

    lastdetection_tt = 0
    while not TERMINATED:

        time.sleep(0.05)

        mainmsglist = []
        for c in cameras:
            mainmsg = "status"
            mainparams = (c.cname, c.frame, c.get_fps(), c.isok, c.active, c.tx)
            if c.active and c.isok:
                try:
                    if c.newframe:
                        # if lag in frames do not do any cnn class.
                        if time.time() - c.tx < 2:
                            c.draw_detections(cnn=True)
                        mainparams = (c.cname, c.frame, c.get_fps(), c.isok, c.active, c.tx)
                        if showframes:
                            cv2.imshow(c.cname, c.frame)
                        c.write_record()
                        new_detections = c.get_new_detections()
                        if new_detections and time.time() - lastdetection_tt > 3:
                            lastdetection_tt = time.time()
                            mainmsg = "detection"
                        c.clear_new_detections()
                except Exception as e:
                    logger.warning(str(e))
            mainmsglist.append((mainmsg, mainparams))

        # send to __main__.py
        pd_outqueue.put(mainmsglist)

        if showframes:
            cv2.waitKey(1) & 0xFF

        try:
            cmd, param = pd_inqueue.get_nowait()
            logger.debug("received " + cmd)
            if cmd == "stop":
                break
            elif cmd == "record on":
                start_all_recordings(cameras)
            elif cmd == "record off":
                stop_all_recordings(cameras)
            elif cmd == "restart_cam":
                cname = param
                logger.info("force restarting camera + " + str(cname))
                restart_cam(cameras, cname)
        except (queue.Empty, EOFError):
            continue
        except Exception:
            continue

    shutdown_cams(cameras)
    clear_all_queues([pd_inqueue, pd_outqueue])
    logger.info("... exited!")
