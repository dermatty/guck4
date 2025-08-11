import configparser
from os.path import expanduser
import os
import shutil
import queue
import json
import subprocess
import sensors
import time
import urllib.request
import platform
import base64
import numpy as np
import datetime
import cv2
import psutil
import paramiko
import torch


class NetConfigReader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config = None

    def get_config(self):
        self.config = None
        config = {}
        try:
            config["ssh_user"] = self.cfg["OPTIONS"]["ssh_user"]
            config["ssh_pass"] = self.cfg["OPTIONS"]["ssh_pass"]
            config["host"] = self.cfg["OPTIONS"]["host"]
            config["ping_freq"] = int(self.cfg["OPTIONS"]["ping_freq"])
        except Exception:
            return None
        config["interfaces"] = []
        idx = 1
        while True:
            try:
                str0 = "INTERFACE" + str(idx)
                name = self.cfg[str0]["name"]
                interface_ip = self.cfg[str0]["interface_ip"]
                gateway_ip = self.cfg[str0]["gateway_ip"]
                gateway_pass = self.cfg[str0]["gateway_pass"]
                pfsense_name = self.cfg[str0]["pfsense_name"]
                reboot_cmd = self.cfg[str0]["reboot_cmd"]
                dns = self.cfg[str0]["dns"]
                idata = {
                    "name": name,
                    "pfsense_name": pfsense_name,
                    "interface_ip": interface_ip,
                    "gateway_ip": gateway_ip,
                    "gateway_pass": gateway_pass,
                    "reboot_cmd": reboot_cmd,
                    "dns": dns
                }
                config["interfaces"].append(idata)
            except Exception:
                break
            idx += 1
        if idx == 1:
            return None
        self.config = config
        return config


class ConfigReader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mscoco_allowed_categories = ["person", "bicycle", "car", "motorcycle", "bus", "truck",
                                          "bird", "cat", "dog"]

    def get_ssh(self):
        ssh_conf = []
        idx = 1
        while True:
            str0 = "SSH" + str(idx)
            try:
                assert self.cfg[str0]["hostname"]
                try:
                    port0 = int(self.cfg[str0]["port"])
                except (Exception, ):
                    port0 = 22
                sshdata = {
                    "hostname": self.cfg[str0]["hostname"],
                    "username": self.cfg[str0]["username"],
                    "port": port0,
                    "idrsa_file": self.cfg[str0]["idrsa_file"],
                    "command": self.cfg[str0]["command"]
                }
                ssh_conf.append(sshdata)
            except Exception:
                break
            idx += 1
        return ssh_conf

    def get_ai(self):
        ai_conf = {}
        try:
            assert self.cfg["AI"]["allowed_categories"]
            ai_conf0 = json.loads(self.cfg["AI"]["allowed_categories"])
            for cat0 in ai_conf0:
                if cat0 in self.mscoco_allowed_categories:
                    try:
                        ai_conf[cat0] = float(ai_conf0[cat0])
                    except Exception:
                        ai_conf[cat0] = 0.9
        except Exception:
            ai_conf = {"person": 0.9}
        if not ai_conf:
            ai_conf = {"person": 0.9}
        try:
            ai_model = self.cfg["AI"]["ai_model"]
        except:
            ai_model = "fasterrcnn_mobilenet_v3_large_320_fpn"
        ai_conf["ai_model"] = ai_model
        return ai_conf

    def get_cameras(self):
        camera_conf = []
        # CAMERA
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
                cdata = {
                        "name": camera_name,
                        "active": active,
                        "stream_url": stream_url,
                        "photo_url": photo_url,
                        "reboot_url": reboot_url,
                        "ptz_mode": ptz_mode,
                        "ptz_right_url": ptz_right_url,
                        "ptz_left_url": ptz_left_url,
                        "ptz_up_url": ptz_up_url,
                        "ptz_down_url": ptz_down_url,
                        "min_area_rect": min_area_rect,
                        "hog_scale": hog_scale,
                        "hog_thresh": hog_thresh,
                        "mog2_sensitivity": mog2_sensitivity,
                        "user": user,
                        "password": password,
                    }
                camera_conf.append(cdata)
            except Exception:
                break
            idx += 1
        return camera_conf

    def get_options(self):
        return self.cfg["OPTIONS"]

    def get_telegram(self):
        return self.cfg["TELEGRAM"]


# setup folders
def setup_dirs(version):
    try:
        install_dir = os.path.dirname(os.path.realpath(__file__))
        userhome = expanduser("~")
        try:
            if version.startswith("3"):
                configfolder = "/.guck3/"
                configfile0 = "guck3.config"
                logsdir = "/media/cifs/dokumente/g3logs/"
            else:
                configfolder = "/.guck4/"
                configfile0 = "guck4.config"
                logsdir = "/media/cifs/dokumente/g4logs/"
        except:
            configfolder = "/.guck4/"
            configfile0 = "guck4.config"
            logsdir = "/media/cifs/dokumente/g4logs/"
        aistatsfile = logsdir + "aistats.txt"
        maindir = userhome + configfolder

        videodir = maindir + "video/"
        photodir = maindir + "photo/"
        configfile = maindir + configfile0
        dirs = {
            "install": install_dir,
            "home": userhome,
            "main": maindir,
            "video": videodir,
            "photo": photodir,
            "logs": logsdir,
            "configfile": configfile,
            "aistatsfile": aistatsfile
        }
    except Exception as e:
        return -1, str(e)

    # check for maindir
    if not os.path.exists(maindir):
        try:
            os.mkdir(maindir)
        except Exception as e:
            return -1, str(e)

    # check for logsdir
    if not os.path.exists(logsdir):
        try:
            os.mkdir(logsdir)
        except Exception as e:
            return -1, str(e) + ": cannot create logs directory!", None, None, None, None, None

    # check for videodir
    if not os.path.exists(videodir):
        try:
            os.mkdir(videodir)
        except Exception as e:
            return -1, str(e) + ": cannot create video directory!", None, None, None, None, None

    # check for photodir
    if not os.path.exists(photodir):
        try:
            os.mkdir(photodir)
        except Exception as e:
            return -1, str(e) + ": cannot create photo directory!", None, None, None, None, None

    # check for configfile
    if not os.path.isfile(configfile):
        config_template = "/etc/default/" + configfile0
        if os.path.isfile(config_template):
            try:
                shutil.copy(config_template, configfile)
            except Exception as e:
                return -1, str(e) + ": cannot initialize " + configfile0 + "!"
        else:
            try:
                shutil.copy(install_dir + "/data/" + configfile0, configfile)
            except Exception as e:
                return -1, str(e) + ": cannot initialize " + configfile0 + "!"

    return 1, dirs


# clear all queues
def clear_all_queues(queuelist):
    for q in queuelist:
        while True:
            try:
                q.get_nowait()
            except (queue.Empty, EOFError):
                break


def check_cfg_file(cfgfile):
    try:
        cfg = configparser.ConfigParser()
        cfg.read(cfgfile)
    except Exception:
        return "error in reading config file!", False
    # USER
    idx = 1
    while True:
        str0 = "USER" + str(idx)
        try:
            assert cfg[str0]["USERNAME"]
            userok = False
            assert cfg[str0]["PASSWORD"]
            userok = True
        except (Exception, ):
            break
        idx += 1
    if idx == 1 or not userok:
        return "error in cfg file [USER]!", False
    # CAMERA
    idx = 1
    while True:
        str0 = "CAMERA" + str(idx)
        try:
            assert cfg[str0]
        except (Exception, ):
            break
        try:
            cameraok = False
            assert cfg[str0]["NAME"] != ""
            assert cfg[str0]["ACTIVE"].lower() in ["yes", "no"]
            assert cfg[str0]["STREAM_URL"] != ""
            assert cfg[str0]["PHOTO_URL"] != ""
            assert cfg[str0]["REBOOT_URL"]
            assert cfg[str0]["PTZ_MODE"].lower() in ["start", "startstop", "none"]
            assert cfg[str0]["PTZ_RIGHT_URL"]
            assert cfg[str0]["PTZ_LEFT_URL"]
            assert cfg[str0]["PTZ_UP_URL"]
            assert cfg[str0]["PTZ_DOWN_URL"]
            assert int(cfg[str0]["MIN_AREA_RECT"]) > 0
            assert float(cfg[str0]["HOG_SCALE"]) > 0
            assert float(cfg[str0]["HOG_THRESH"]) > 0
            assert float(cfg[str0]["MOG2_SENSITIVITY"])
            cameraok = True
        except (Exception, ):
            break
        idx += 1
    if idx == 1 or not cameraok:
        return "error in cfg file [CAMERA]!", False
    # OPTIONS
    try:
        assert cfg["OPTIONS"]["REDIS_HOST"].strip() != ""
    except (Exception, ):
        return "error in cfg file [OPTIONS][REDIS_HOST]!", False
    try:
        assert int(cfg["OPTIONS"]["REDIS_PORT"]) > 0
    except (Exception, ):
        return "error in cfg file [OPTIONS][REDIS_PORT]!", False
    try:
        assert cfg["OPTIONS"]["KEYBOARD_ACTIVE"].lower() in ["yes", "no"]
    except (Exception, ):
        return "error in cfg file [OPTIONS][KEYBOARD_ACTIVE]!", False
    try:
        assert cfg["OPTIONS"]["LOGLEVEL"].lower() in ["debug", "info", "warning", "error"]
    except (Exception, ):
        return "error in cfg file [OPTIONS][LOGLEVEL]!", False
    try:
        assert cfg["OPTIONS"]["SHOWFRAMES"].lower() in ["yes", "no"]
    except (Exception, ):
        return "error in cfg file [OPTIONS][SHOWFRAMES]!", False
    # no check for ["OPTIONS"]["ADDTL_PHOTO_PATH"] cause it iss optional
    # TELEGRAM
    try:
        tgram_active = True if cfg["TELEGRAM"]["ACTIVE"].lower() == "yes" else False
    except (Exception, ):
        return "error in cfg file [TELEGRAM][ACTIVE]!", False
    if tgram_active:
        try:
            assert cfg["TELEGRAM"]["TOKEN"]
        except (Exception, ):
            return "error in cfg file [TELEGRAM][TOKEN]!", False
        try:
            chatids = json.loads(cfg.get("TELEGRAM", "CHATIDS"))
            if not isinstance(chatids, list):
                return "error in cfg file [TELEGRAM][CHATIDS]!", False
        except (Exception, ):
            return "error in cfg file [TELEGRAM][CHATIDS]!", False
    return "", True


def get_external_ip(hostlist=[("WAN2TMO_DHCP", "raspisens"), ("WAN_DHCP", "etec")]):
    procstr = 'curl https://api.ipdata.co/"$(dig +short myip.opendns.com @resolver1.opendns.com)"'
    procstr += "?api-key=b8d4413e71b0e5827c4624c856f0439ee6b64ff8a71c419bfcd2d14c"

    iplist = []
    for gateway, hostn in hostlist:
        try:
            ssh = subprocess.Popen(["ssh", hostn, procstr], shell=False, stdout=subprocess.PIPE, stderr=subprocess. PIPE)
            sshres = ssh.stdout.readlines()
            s0 = ""
            for ss in sshres:
                s0 += ss.decode("utf-8")
            d = json.loads(s0)
            iplist.append((gateway, hostn, d["ip"], d["asn"]["name"]))
        except (Exception, ):
            iplist.append((gateway, hostn, "N/A", "N/A"))
    return iplist


def get_ssh_results(state_data):
    reslist = []
    for s in state_data.SSH_CONFIG:
        try:
            hostname = s["hostname"]
            username = s["username"]
            idrsa_file = s["idrsa_file"],
            command = s["command"]
            port = s["port"]
            command0 = command
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(hostname, username=username, key_filename=idrsa_file, timeout=10, port=port)
            stdin, stdout, stderr = ssh_client.exec_command(command)
            res0 = stdout.readlines()
            res00 = []
            
            if "webstatus" in command.lower():
                command0 = "webstatus"
                for r0 in res0:
                    tempstr0 = r0.split(" / ")
                    res00.append(tempstr0[1] + " is " + tempstr0[2].rstrip() + " (" + tempstr0[0] + ")")
                    
            elif "wip2" in command.lower():
                command0 = "whatismyip2"
                for r0 in res0:
                    tempstr0 = r0.split(": ")
                    interface0 = tempstr0[1].split(" via ")[0]
                    ip0 = tempstr0[2]
                    provider0 = " ".join(tempstr0[3].split(" ")[1:])
                    country0 = tempstr0[4].rstrip()
                    res00.append(interface0 + ": " + ip0 + " (" + provider0 + "/" +  country0 + ")")
                    
            elif command.lower() == "/home/stephan/sens":
                command0 = "sens"
                tempstr = "-/-"
                hum = "-/-"
                cpu_temp = "-/-"
                for r0 in res0:
                    try:
                        tempstr0 = r0.split(" ")
                        tempstr = str(round(float(tempstr0[2]), 1)) + "°C"
                        hum = str(round(float(tempstr0[3]), 1)) + "%"
                        cpu_temp = str(round(float(tempstr0[4]), 1)) + "°C"
                    except (Exception, ):
                        pass
                    res00.append("room: " + tempstr + " / hum.: " + hum + " / cpu: " + cpu_temp)
                    break

            elif command.lower() == "temp":
                command0 = "temp"
                n0 = 0
                sum0 = 0
                for r0 in res0:
                    if "dev.cpu" in r0:
                        try:
                            tempstr = r0.split(" ")[1]
                            sum0 += float("".join(s1 for s1 in tempstr if s1.isdigit() or s1 == "."))
                            n0 += 1
                        except (Exception, ):
                            pass
                try:
                    sum0 = sum0 / n0
                except (Exception, ):
                    sum0 = 0
                res00.append("cpu: " + str(round(sum0, 1)) + "°C")

            res0list = [hostname, command0, res00]
            reslist.append(res0list)
            ssh_client.close()
        # except (Exception, ):
        except Exception as e:
            print(str(e))
    return reslist

"""def get_sens_temp(hostn="raspisens", filen="/home/pi/sens.txt"):
    procstr = "cat " + filen
    ssh = subprocess.Popen(["ssh", hostn, procstr], shell=False, stdout=subprocess.PIPE, stderr=subprocess. PIPE)
    sshres = ssh.stdout.readlines()
    n = 0
    temp = 0
    hum = 0
    for s in sshres:
        try:
            s0 = s.decode("utf-8").split(" ")
            temp += float(s0[2])
            hum += float(s0[3])
        except Exception:
            pass
        n += 1
    if n > 0:
        temp = temp / n
        hum = hum / n
    return temp, hum"""


def check_cam_health(state_data):
    cam_health = {}
    try:
        for c in state_data.CAMERADATA:
            cname, cframe, c_fps, cisok, cactive, ctx = c
            if not cactive:
                c_status = "DISABLED"
                dt = -1
            else:
                try:
                    dt = time.time() - ctx
                except (Exception, ):
                    dt = 31
                if dt > 30 or not cisok:
                    c_status = "DOWN"
                elif dt > 3:
                    c_status = "DELAYED"
                else:
                    c_status = "RUNNING"
            cam_health[cname] = {"status": c_status, "fps": c_fps, "dt": dt}
    except (Exception, ):
        pass
    return cam_health


def get_status(state_data, version):
    osversion = os.popen("cat /etc/os-release").read().split("\n")[2].split("=")[1].replace('"', '')
    processor = platform.processor().lower()

    # os & version
    ret = "------- General -------"
    ret += "\nOS: " + osversion
    ret += "\nVersion: " + version
    ret += "\nAlarm System Active: "
    ret += "YES" if state_data.PD_ACTIVE else "NO"
    ret += "\n------- System -------"

    # memory
    overall_mem = round(psutil.virtual_memory()[0] / float(2 ** 20) / 1024, 2)
    free_mem = round(psutil.virtual_memory()[1] / float(2 ** 20) / 1024, 2)
    used_mem = round(overall_mem - free_mem, 2)
    perc_used = round((used_mem / overall_mem) * 100, 2)
    mem_crit = False
    if perc_used > 85:
        mem_crit = True

    # cpu
    cpu_perc0 = [cpuload/100 for cpuload in psutil.cpu_percent(interval=0.25, percpu=True)]
    cpu_avg = sum(cpu_perc0)/float(len(cpu_perc0))
    cpu_perc = (max(cpu_perc0) * 0.7 + cpu_avg * 0.3)/2

    if "intel" in processor:
        cpu_vendor = "intel"
        cpu_crit_value = 70.0
    elif "amd" in processor:
        cpu_vendor = "amd"
        cpu_crit_value = 85.0
    else:
        cpu_vendor = "unknown"
        cpu_crit_value = 85.0

    ret += "\nRAM: " + str(perc_used) + "% (" + str(used_mem) + " GB) of overall " + str(overall_mem) + \
           " GB used"
    ret += "\nCPU info: " + platform.processor() + " / critical @ " + str(cpu_crit_value) + "°C"
    ret += "\nCPU load: Avg=" + str(int(cpu_avg*100)) + "% / Max=" + str(int(max(cpu_perc0)*100)) + "%"

    # sensors / cpu temp
    sensors.init()
    cpu_temp = []
    for chip in sensors.iter_detected_chips():
        for feature in chip:
            if ((cpu_vendor == "intel" and feature.label[0:4] == "Core") or
                    ("amd" in processor and feature.label[0:4] in ["Tctl", "Tccd"])):
                temp0 = feature.get_value()
                cpu_temp.append(temp0)
                # ret += "\nCPU " + feature.label + " temp.: " + str(round(temp0, 2)) + "°C"

    sensors.cleanup()
    if len(cpu_temp) > 0:
        avg_cpu_temp = sum(c for c in cpu_temp)/len(cpu_temp)
        max_cpu_temp = max(cpu_temp)
    else:
        avg_cpu_temp = 0.0
        max_cpu_temp = 0.0

    ret += "\nCPU temp: Avg=" + str(round(avg_cpu_temp, 1)) + "°C" + " / Max=" + str(round(max_cpu_temp, 1)) + "°C"
    if avg_cpu_temp > cpu_crit_value*0.8 and max_cpu_temp > cpu_crit_value:
        cpu_crit = True
    else:
        cpu_crit = False


    # gpu
    smifn = None
    gpu_name = "N/A"
    gputemp_str = "0.0"
    gpuutil_str = "0.0%"
    if os.path.exists("/opt/bin/nvidia-smi"):
        smifn = "/opt/bin/nvidia-smi"
    elif os.path.exists("/usr/bin/nvidia-smi"):
        smifn = "/usr/bin/nvidia-smi"
    if smifn:
        try:
            gpu_name = subprocess.Popen([smifn, "--query-gpu=gpu_name", "--format=csv"],
                                       stdout=subprocess.PIPE).stdout.readlines()[1].decode()[:-1]
            gputemp = subprocess.Popen([smifn, "--query-gpu=temperature.gpu", "--format=csv"],
                                       stdout=subprocess.PIPE).stdout.readlines()[1]
            gpuutil = subprocess.Popen([smifn, "--query-gpu=utilization.gpu", "--format=csv"],
                                       stdout=subprocess.PIPE).stdout.readlines()[1]
            gputemp_str = gputemp.decode("utf-8").rstrip()
            gpuutil_str = gpuutil.decode("utf-8").rstrip()
        except Exception as e:
            pass
    ret += "\nGPU: " + gpu_name + " / " + gputemp_str + "°C" + " / " + gpuutil_str + " util."
    try:
        if float(gputemp_str) > 70.0:
            gpu_crit = True
        else:
            gpu_crit = False
    except Exception:
        gpu_crit = False

    cam_crit = False
    if state_data.PD_ACTIVE:
        ret += "\n------- Cameras -------"
        for c in state_data.CAMERADATA:
            cname, cframe, cfps, cisok, cactive, ctx = c
            if not cactive:
                ctstatus0 = "DISABLED"
                ret += "\n" + cname + " " + ctstatus0
            else:
                try:
                    dt = time.time() - ctx
                except Exception:
                    dt = 31
                if dt > 30 or not cisok:
                    ctstatus0 = "DOWN"
                elif dt > 3:
                    ctstatus0 = "DELAYED"
                else:
                    ctstatus0 = "running"
                    if ctstatus0 in ["DOWN", "DELAYED"]:
                        cam_crit = True
                    else:
                        cam_crit = False
                ret += "\n" + cname + " " + ctstatus0 + " @ %3.1f fps" % cfps + ", (%.2f" % dt + " sec. ago)"

    ssh_reslist = get_ssh_results(state_data)
    for sshr in ssh_reslist:
        ssh_hostname, ssh_command, ssh_res0 = sshr
        ret += "\n--- " + str(ssh_hostname) + " > " + str(ssh_command)
        for l0 in ssh_res0:
            ret += "\n   " + l0
    ret += "\n------- System Summary -------"
    ret += "\nRAM: "
    ret += "CRITICAL!" if mem_crit else "OK!"
    ret += "\nCPU: "
    ret += "CRITICAL!" if cpu_crit else "OK!"
    ret += "\nGPU: "
    ret += "CRITICAL!" if gpu_crit else "OK!"
    ret += "\nCAMs: "
    if state_data.PD_ACTIVE:
        ret += "CRITICAL!" if cam_crit else "OK!"
    else:
        ret += "NOT RUNNING!"
    return ret, mem_crit, cpu_crit, gpu_crit, cam_crit


def get_free_photos(dir, camera_config, logger):
    freedir = dir + "free/"
    if not os.path.exists(freedir):
        try:
            os.mkdir(freedir)
        except Exception as e:
            logger.warning(str(e))
            return []
    filelist = [f for f in os.listdir(freedir)]
    for f in filelist:
        try:
            os.remove(freedir + f)
        except Exception as e:
            logger.warning(str(e))
    urllist = [(c["name"], c["photo_url"], c["user"], c["password"]) for c in camera_config]
    freephotolist = []
    for cname, url, user, pw in urllist:
        try:
            request = urllib.request.Request(url)
            base64string = base64.b64encode(bytes('%s:%s' % (user, pw), 'ascii'))
            request.add_header("Authorization", "Basic %s" % base64string.decode('utf-8'))
            result = urllib.request.urlopen(request, timeout=3)
            image = np.asarray(bytearray(result.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            datestr = datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S")
            photoname = freedir + cname + "_" + datestr + ".jpg"
            cv2.imwrite(photoname, image)
            freephotolist.append(photoname)
        except Exception as e:
            logger.warning(str(e))
    return freephotolist
