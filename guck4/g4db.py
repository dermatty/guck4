from .mplogging import whoami
import pickle
import time
from guck4 import __appname__, __appabbr__


# --------------- REDIS API -------------------
class RedisAPI:
    def __init__(self, red, dirs, cfg, logger):
        self.red = red
        self.cfg = cfg
        self.dirs = dirs
        self.logger = logger
        self.copyok = True
        if not self.getp(__appabbr__ + "_userdata"):
            self.setp(__appabbr__ + "_userdata", {})
        if not self.getp(__appabbr__ + "_photodata"):
            self.setp(__appabbr__ + "_photodata", [])
        if not self.getp(__appabbr__ + "_userdata_last_updated"):
            self.setp(__appabbr__ + "_userdata_last_updated", 0)
        if not self.getp(__appabbr__ + "_new_detections"):
            self.setp(__appabbr__ + "_new_detections", 0)
        if not self.getp(__appabbr__ + "_hoststatus"):
            self.setp(__appabbr__ + "_hoststatus", None)
        if not self.getp(__appabbr__ + "_free_photodata"):
            self.setp(__appabbr__ + "_free_photodata", [])
        if not self.getp(__appabbr__ + "_free_photodata_status"):
            self.setp(__appabbr__ + "_free_photodata_status", [])
        self.copy_users_to_redis()
        self.copy_cameras_to_redis()
        self.setp(__appabbr__ + "_putcmd", "")

    def setp(self, key, value):
        try:
            ret = self.red.set(key, pickle.dumps(value))
            return ret
        except Exception:
            return False

    def getp(self, key):
        try:
            ret = pickle.loads(self.red.get(key))
            return ret
        except Exception:
            return None

    def set_host_status(self, status):
        self.setp(__appabbr__ + "_hoststatus", status)

    def get_host_status(self):
        return self.getp(__appabbr__ + "_hoststatus")

    def set_free_photodata(self, data):
        self.setp(__appabbr__ + "_free_photodata", data)

    def get_free_photodata(self):
        return self.getp(__appabbr__ + "_free_photodata")

    def set_putcmd(self, cmd):
        self.setp(__appabbr__ + "_putcmd", cmd)

    def get_putcmd(self):
        ret = self.getp(__appabbr__ + "_putcmd")
        self.setp(__appabbr__ + "_putcmd", "")
        return ret

    def copy_redis_to_cameras_cfg(self):
        self.logger.debug(whoami() + "copying redis camera data to config ...")
        cameras = self.getp(__appabbr__ + "_cameras")
        for i, c in enumerate(cameras, start=1):
            cstr = "CAMERA" + str(i)
            self.cfg[cstr]["ACTIVE"] = "yes" if c[0] else "no"
            self.cfg[cstr]["NAME"] = c[1]
            self.cfg[cstr]["STREAM_URL"] = c[2]
            self.cfg[cstr]["PHOTO_URL"] = c[3]
            self.cfg[cstr]["REBOOT_URL"] = c[4]
            self.cfg[cstr]["PTZ_MODE"] = c[5]
            self.cfg[cstr]["PTZ_RIGHT_URL"] = c[6]
            self.cfg[cstr]["PTZ_LEFT_URL"] = c[7]
            self.cfg[cstr]["PTZ_UP_URL"] = c[8]
            self.cfg[cstr]["PTZ_DOWN_URL"] = c[9]
            self.cfg[cstr]["MIN_AREA_RECT"] = str(c[10])
            self.cfg[cstr]["HOG_SCALE"] = str(c[11])
            self.cfg[cstr]["HOG_THRESH"] = str(c[12])
            self.cfg[cstr]["MOG2_SENSITIVITY"] = str(c[13])
            self.cfg[cstr]["USER"] = str(c[14])
            self.cfg[cstr]["PASSWORD"] = str(c[15])
        # write to cfg_file
        cfg_file = self.dirs["configfile"]
        try:
            with open(cfg_file, "w") as f:
                self.cfg.write(f)
        except Exception as e:
            self.logger.error(whoami() + str(e) + ", cannot write redis to config file!")
            return -1
        self.logger.debug(whoami() + "... redis camera data copied to config!")
        return 1

    def copy_cameras_to_redis(self):
        self.logger.debug(whoami() + "copying camera data to redis ...")
        self.setp(__appabbr__ + "_cameras", [])
        cameralist = []
        idx = 1
        while True:
            str0 = "CAMERA" + str(idx)
            try:
                assert self.cfg[str0]["NAME"]
                active = True if self.cfg[str0]["ACTIVE"].lower() == "yes" else False
                camera_name = self.cfg[str0]["NAME"]
                stream_url = self.cfg[str0]["STREAM_URL"]
                photo_url = self.cfg[str0]["PHOTO_URL"]
                user = self.cfg[str0]["USER"]
                password = self.cfg[str0]["PASSWORD"]
                reboot_url = self.cfg[str0]["REBOOT_URL"]
                ptz_mode = self.cfg[str0]["PTZ_MODE"].lower()
                if ptz_mode not in ["start", "startstop", "none"]:
                    ptz_mode = "none"
                ptz_right_url = self.cfg[str0]["PTZ_RIGHT_URL"]
                ptz_left_url = self.cfg[str0]["PTZ_LEFT_URL"]
                ptz_up_url = self.cfg[str0]["PTZ_UP_URL"]
                ptz_down_url = self.cfg[str0]["PTZ_DOWN_URL"]
                min_area_rect = int(self.cfg[str0]["MIN_AREA_RECT"])
                hog_scale = float(self.cfg[str0]["HOG_SCALE"])
                hog_thresh = float(self.cfg[str0]["HOG_THRESH"])
                mog2_sensitivity = float(self.cfg[str0]["MOG2_SENSITIVITY"])
                cameralist.append((active, camera_name, stream_url, photo_url, reboot_url, ptz_mode, ptz_right_url,
                                  ptz_left_url, ptz_up_url, ptz_down_url, min_area_rect, hog_scale, hog_thresh,
                                  mog2_sensitivity, user, password))
            except Exception:
                break
            idx += 1
        if idx == 1:
            self.copyok = False
            return
        self.setp(__appabbr__ + "_cameras", cameralist)
        self.logger.debug(whoami() + " ... camera data copied to redis!")

    def get_cameras(self):
        camera_conf = []
        cameralist = self.getp(__appabbr__ + "_cameras")
        for c in cameralist:
            cdata = {
                    "name": c[1],
                    "active": c[0],
                    "stream_url": c[2],
                    "photo_url": c[3],
                    "reboot_url": c[4],
                    "ptz_mode": c[5],
                    "ptz_right_url": c[6],
                    "ptz_left_url": c[7],
                    "ptz_up_url": c[8],
                    "ptz_down_url": c[9],
                    "min_area_rect": c[10],
                    "hog_scale": c[11],
                    "hog_thresh": c[12],
                    "mog2_sensitivity": c[13],
                    "user": c[14],
                    "password": c[15]
                }
            camera_conf.append(cdata)
        if not camera_conf:
            return None
        return camera_conf

    def copy_users_to_redis(self):
        self.setp(__appabbr__ + "_users", {})
        idx = 1
        users = {}
        while True:
            str0 = "USER" + str(idx)
            try:
                username = self.cfg[str0]["USERNAME"]
                password = self.cfg[str0]["PASSWORD"]
                users[username] = password
            except Exception:
                break
            idx += 1
        if idx == 1:
            self.copyok = False
            return
        self.setp(__appabbr__ + "_users", users)
        self.logger.debug(whoami() + "user data copied to db")

    def get_photodata(self):
        return self.getp(__appabbr__ + "_photodata")

    def insert_photodata(self, photonames):
        photodata = self.getp(__appabbr__ + "_photodata")
        try:
            for p in photonames:
                photodata.insert(0, p)
                if len(photodata) > 50:
                    del photodata[-1]
            self.setp(__appabbr__ + "_photodata", photodata)
        except Exception as e:
            self.logger.error(whoami() + str(e) + ", cannot insert photodata!")

    def get_users(self):
        return self.getp(__appabbr__ + "_users")

    def get_userdata(self):
        return self.getp(__appabbr__ + "_userdata")

    def get_userdata_last_updated(self):
        return self.getp(__appabbr__ + "_userdata_last_updated")

    def user_in_userdata(self, username):
        userdata = self.getp(__appabbr__ + "_userdata")
        return userdata, (len([1 for key in userdata if key == username]) > 0)

    def insert_update_userdata(self, username, lasttm, active, no_newdetections, photolist):
        try:
            userdata, user_in_userdata = self.user_in_userdata(username)
            if not user_in_userdata:
                userdata[username] = {}
            userdata[username]["lastttm"] = lasttm
            userdata[username]["active"] = active
            userdata[username]["no_newdetections"] = no_newdetections
            userdata[username]["photolist"] = photolist
            self.setp(__appabbr__ + "_userdata", userdata)
            self.setp(__appabbr__ + "_userdata_last_updated", time.time())
        except Exception as e:
            self.logger.warning(whoami() + str(e))
