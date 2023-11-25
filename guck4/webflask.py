from __future__ import unicode_literals
import multiprocessing
import gunicorn.app.base
import os
from flask import Flask, render_template, request, g, redirect, url_for
from flask.logging import default_handler
from flask_sse import sse
from flask_session import Session
import flask_login
from setproctitle import setproctitle
from .mplogging import whoami
from .g4db import RedisAPI
import time
from guck4 import models, setup_dirs, check_cfg_file, __version__, __appname__, __appabbr__, __startmode__
from threading import Thread, Lock
import logging
import redis
import configparser
import signal
import shutil, sys
import html2text


RED = None
USERS = None
DIRS = None
maincomm = None

# get redis data
ret, dirs = setup_dirs(__version__)
cfg_file = dirs["configfile"]
static_dir = '/'.join(__file__.split("/")[:-1]) + "/static/"


cfg = configparser.ConfigParser()
cfg.read(cfg_file)
try:
    REDIS_HOST = cfg["OPTIONS"]["REDIS_HOST"]
except Exception:
    REDIS_HOST = "127.0.0.1"
try:
    REDIS_PORT = int(cfg["OPTIONS"]["REDIS_PORT"])
except Exception:
    REDIS_PORT = 6379


REDISCLIENT = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)


# -------------- Helper functions --------------

def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1


def sighandler(a, b):
    try:
        # RED.copy_redis_to_cameras_cfg()
        filelist = [f for f in os.listdir(static_dir) if f.endswith(".jpg")]
        for f in filelist:
            os.remove(static_dir + f)
    except Exception as e:
        print(str(e))


# -------------- Init Flask App --------------
app = Flask(__name__)
app.secret_key = "dfdsmdsv11nmDFSDfds_ers"
app.config["REDIS_URL"] = "redis://" + REDIS_HOST + ":" + str(REDIS_PORT)
app.config['SESSION_TYPE'] = "redis"
app.config["SESSION_REDIS"] = REDISCLIENT
app.register_blueprint(sse, url_prefix='/stream')
Session(app)


# -------------- MainCommunicator --------------
class MainCommunicator(Thread):

    def __init__(self, inqueue, outqueue, app, red):
        Thread.__init__(self)
        self.daemon = True
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.lock = Lock()
        self.app = app
        self.red = red
        self.userdata_updated = False
        self.no_detections = 0

    def sse_publish(self):
        if self.pd_active:
            with self.app.app_context():
                result0 = render_template("guckphoto.html", nralarms=self.no_detections, guckstatus="on")
                type0 = "nrdet0"
                sse.publish({"message": result0}, type=type0)
                type0 = "title0"
                sse.publish({"message": str(self.no_detections)}, type=type0)
        else:
            with self.app.app_context():
                result0 = render_template("guckphoto.html", nralarms=self.no_detections, guckstatus="off")
                type0 = "nrdet0"
                sse.publish({"message": result0}, type=type0)
                type0 = "title0"
                sse.publish({"message": str(self.no_detections)}, type=type0)
        self.last_sse_published = time.time()

    def run(self):
        self.pd_active = "N/A"
        self.last_sse_published = 0
        while True:
            time.sleep(0.5)
            try:
                with self.lock:
                    pcmd = self.red.get_putcmd()
                    if pcmd == "pdstart":
                        self.outqueue.put(("set_pdstart", None))
                        cmd, data = self.inqueue.get()
                    elif pcmd == "pdrestart":
                        self.outqueue.put(("set_pdrestart", None))
                        cmd, data = self.inqueue.get()
                    elif pcmd == "pdstop":
                        self.outqueue.put(("set_pdstop", None))
                        cmd, data = self.inqueue.get()
                    elif pcmd == "get_host_status":
                        self.outqueue.put(("get_host_status", None))
                        cmd, data = self.inqueue.get()
                        self.red.set_host_status(data)
                    elif pcmd == "get_free_photodata":
                        self.outqueue.put(("get_free_photodata", None))
                        cmd, data = self.inqueue.get()
                        self.red.set_free_photodata(data)
                    else:
                        self.outqueue.put(("get_pd_status", None))
                        cmd, data = self.inqueue.get()
                        try:
                            new_detections = len(data)
                        except Exception:
                            new_detections = 0
            except Exception:
                pass
            try:
                lastuserdata_tt = self.red.get_userdata_last_updated()
                userdata_updated_since = (lastuserdata_tt > self.last_sse_published)
            except Exception:
                userdata_updated_since = False
            if cmd != self.pd_active or userdata_updated_since or new_detections > 0:
                if new_detections > 0:
                    self.red.insert_photodata(data)
                self.pd_active = cmd
                self.no_detections += new_detections
                self.sse_publish()


# -------------- Login Manager --------------

login_manager = flask_login.LoginManager()
login_manager.login_view = 'userlogin'
login_manager.init_app(app)


@app.before_request
def beforerequest():
    try:
        user0 = flask_login.current_user.get_id()
        g.user = user0
        if user0 is not None:
            userdata, user_in_userdata = RED.user_in_userdata(user0)
            if not userdata or not user_in_userdata:
                RED.insert_update_userdata(user0, time.time(), True, 0, [])
            else:
                RED.insert_update_userdata(user0, time.time(), True, userdata[user0]["no_newdetections"],
                                           userdata[user0]["photolist"])
    except Exception as e:
        app.logger.info(whoami() + str(e))
        pass


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(email):
    if email not in USERS:
        return
    try:
        user = User()
        user.id = email
    except Exception as e:
        app.logger.warning(whoami() + str(e))
    return user


@app.route("/userlogout", methods=['GET', 'POST'])
@flask_login.login_required
def userlogout():
    userid = flask_login.current_user.get_id()
    app.logger.info(whoami() + ": user logging out - " + userid)
    flask_login.logout_user()
    return redirect(url_for("index"))


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == "GET":
        app.logger.info(whoami() + ": new user starting to log in ...")
        userloginform = models.UserLoginForm(request.form)
        return render_template("login.html", userloginform=userloginform, userauth=flask_login.current_user.is_authenticated)
    else:
        userloginform = models.UserLoginForm(request.form)
        email = userloginform.email.data
        pw = userloginform.password.data
        app.logger.info(whoami() + ": user trying to log in - " + str(email) + " from ip " + str(request.remote_addr))
        try:
            correct_pw = USERS[email]
        except Exception:
            app.logger.warning(whoami() + ": user log in failed - " + str(email) + " from ip " + str(request.remote_addr))
            return redirect(url_for("index"))
        if pw == correct_pw:
            app.logger.info(whoami() + ": user logged in - " + email)
            try:
                user = User()
                user.id = email
                flask_login.login_user(user)
            except Exception:
                pass
            return render_template("index.html")
        app.logger.warning(whoami() + ": user log in failed - " + str(email) + " from ip " + str(request.remote_addr))
        return redirect(url_for('index'))


# -------------- Index.html / Home --------------

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
@flask_login.login_required
def index():
    return render_template("index.html")


# -------------- config --------------

@app.route("/config", methods=['GET', 'POST'])
@flask_login.login_required
def config():
    status = ""
    config_file = DIRS["configfile"]
    if request.method == "GET":
        with open(config_file, "r") as f:
            content = ""
            for line in f:
                content += (line + "<br>")
    elif request.method == 'POST':
        h = html2text.HTML2Text()
        h.body_width = 0
        content = h.handle(request.form.get('editordata'))
        if request.form['submit'] == 'cancel':
            return redirect(url_for('index'))
        elif request.form['submit'] == "submit" and content != "":
            try:
                temp_config_file = "config_file" + ".bak"
                with open(temp_config_file, "w") as f:
                    f.write(content)
                errmsg, cfg_file_ok = check_cfg_file(temp_config_file)
                if not cfg_file_ok:
                    status = errmsg
                else:
                    try:
                        with open(config_file, "w") as f:
                            f.write(content)
                        os.remove(temp_config_file)
                        status = "config file saved!"
                    except Exception:
                        status = "cannot write config file!"
            except Exception:
                status = "cannot check config file (write error)!"
        else:
            # indicate a failure to enter required data
            status = 'ERROR: page title and content are required!'
        with open(config_file, "r") as f:
            content = ""
            for line in f:
                content += (line + "<br>")

    return render_template('configedit.html', content=content, config_file=config_file, status=status)


# -------------- pd_start --------------
@app.route("/pdstart", methods=['GET', 'POST'])
@flask_login.login_required
def pdstart():
    RED.set_putcmd("pdstart")
    return render_template("start.html")


# -------------- pd_stop --------------
@app.route("/pdstop", methods=['GET', 'POST'])
@flask_login.login_required
def pdstop():
    RED.set_putcmd("pdstop")
    return render_template("stop.html")

# -------------- status --------------
@app.route("/status", methods=['GET', 'POST'])
@flask_login.login_required
def status():
    RED.set_host_status(None)
    RED.set_putcmd("get_host_status")
    host_status = None
    while not host_status:
        host_status = RED.get_host_status()
        if not host_status:
            time.sleep(0.05)
    statuslist, mem_crit, cpu_crit, gpu_crit, cam_crit = host_status
    statuslist = statuslist.split("\n")
    return render_template("status.html", statuslist=statuslist)

# -------------- restart --------------
@app.route("/pdrestart", methods=['GET', 'POST'])
@flask_login.login_required
def restart():
    RED.set_putcmd("pdrestart")
    return render_template("restart.html")

# -------------- detections --------------
@app.route("/detections", methods=['GET', 'POST'])
@flask_login.login_required
def detections():
    filelist = [f for f in os.listdir(static_dir) if f.endswith(".jpg")]
    for f in filelist:
        os.remove(static_dir + f)
    detlist = []
    for p in RED.get_photodata():
        try:
            p1 = os.path.basename(p)
            shutil.copy(p, static_dir + p1)
            detlist.append(p1)
        except Exception:
            pass
    return render_template('detections.html', detlist=detlist)


# -------------- photos --------------
@app.route("/photos", methods=['GET', 'POST'])
@flask_login.login_required
def photos():
    RED.set_free_photodata(None)
    RED.set_putcmd("get_free_photodata")
    free_photodata = None
    while not free_photodata:
        free_photodata = RED.get_free_photodata()
        if not free_photodata:
            time.sleep(0.5)
    detlist = []
    for p in free_photodata:
        try:
            p1 = os.path.basename(p)
            shutil.copy(p, static_dir + p1)
            detlist.append(p1)
        except Exception:
            pass
        pass
    return render_template('photos.html', detlist=detlist)


# -------------- StandaloneApplication/main --------------


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def main(cfg, dirs, inqueue, outqueue, loggerqueue):
    global RED
    global USERS
    global DIRS
    global app
    global maincomm
    global static_dir

    setproctitle(__appabbr__ + "." + os.path.basename(__file__))

    DIRS = dirs

    log_handler = logging.FileHandler(dirs["logs"] + "webflask.log", mode="w")
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    app.logger.removeHandler(default_handler)
    app.logger.setLevel(logging.DEBUG)
    app.logger.addHandler(log_handler)

    app.logger.info(whoami() + "starting ...")
    app.logger.info(whoami() + "static dir = " + static_dir)

    RED = RedisAPI(REDISCLIENT, dirs, cfg, app.logger)
    if not RED.copyok:
        app.logger.error(whoami() + ": cannot init redis, exiting")
        outqueue.put("False")
    else:
        outqueue.put("True")

    USERS = RED.get_users()

    # start communicator thread
    maincomm = MainCommunicator(inqueue, outqueue, app, RED)
    maincomm.start()

    try:
        certfile = cfg["OPTIONS"]["CERTFILE"]
        keyfile = cfg["OPTIONS"]["KEYFILE"]
        options = {
            'bind': '%s:%s' % ('0.0.0.0', '8000'),
            'certfile': certfile,
            'keyfile': keyfile,
            'capture_output': True,
            'debug': True,
            'graceful_timeout': 10,
            "timeout": 120,
            # 'worker_class': 'gevent',
            'worker_class': 'gthread',
            'workers': 3,
            "threads": 3,
            "worker-connections": 100
        }
        app.logger.info(whoami() + ": binding gunicorn to https!")
    except Exception:
        options = {
            'bind': '%s:%s' % ('0.0.0.0', '8000'),
            'capture_output': True,
            'debug': True,
            'graceful_timeout': 10,
            "timeout": 120,
            #'worker_class': 'gevent',
            'worker_class': 'gthread',
            'workers': 3,
            "threads": 3,
            "worker-connections": 100
        }
        app.logger.warning(whoami() + ": binding gunicorn to http!")
    signal.signal(signal.SIGFPE, sighandler)     # nicht die feine englische / faut de mieux
    StandaloneApplication(app, options).run()


if __name__ == '__main__':
    main()
