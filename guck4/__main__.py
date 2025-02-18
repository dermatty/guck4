from threading import Thread
import psutil, os, datetime, sys, configparser, signal, time, queue, requests, json
from setproctitle import setproctitle
import logging, logging.handlers
import multiprocessing as mp
import cv2

import fridagram as fg

from guck4 import setup_dirs, mplogging, peopledetection, clear_all_queues, ConfigReader, check_cam_health
from guck4 import get_status, get_free_photos, webflask, __version__, __startmode__, __appname__, __appabbr__

from .mplogging import whoami


TERMINATED = False
RESTART = False
_HEARTBEAT_FRQ = 10  # heartbeat frequency in minutes
_CLEAR_BOT_FREQ = 6  # clear bot freq in hours


def GeneralMsgHandler(msg, bot, state_data):
	global TERMINATED
	global RESTART
	# bot = tgram / kbd
	bot0 = bot.lower()
	if bot0 not in ["tgram", "kbd", "wf"]:
		return None
	reply = ""
	if msg.startswith("start"):
		try:
			aiparam = msg.split("start")[1].lstrip()
		except (Exception, ):
			aiparam = None
		state_data.MAINQUEUE.put(("start", aiparam))
		reply = "starting " + __appname__ + " people detection ..."
	elif msg == "photos":
		if bot0 == "tgram":
			reply = "collecting photo snapshots from all cameras ..."
		elif bot0 == "kbd":
			reply = "cannot send photos in text console!"
	elif msg == "stop":
		if state_data.mpp_peopledetection:
			if state_data.mpp_peopledetection.pid:
				state_data.MAINQUEUE.put(("stop", None))
				reply = "stopping " + __appname__ + " people detection ..."
		else:
			reply = __appname__ + " people detection is NOT running, cannot stop!"
	elif msg == "exit!!" or msg == "restart!!":
		if msg == "restart!!":
			reply = "restarting " + __appname__ + "!"
		else:
			reply = "exiting " + __appname__ + "!"
		state_data.MAINQUEUE.put((msg, None))
	elif msg == "status":
		reply, _, _, _, _ = get_status(state_data, __version__)
	elif msg == "?" or msg == "help":
		reply = "start|stop|exit!!|restart!!|status|photos|camrestart <n>"
	elif msg.startswith("camrestart"):
		try:
			camnr = int(msg.split("camrestart")[1])
			if camnr not in range(0, len(state_data.CAMERADATA)-1):
				raise Exception("Camera # out of valid range!")
			reply = "restarting camera  #" + str(camnr)
			state_data.MAINQUEUE.put(("camrestart", camnr))
		except Exception as e:
			reply = str(e) + " :cannot restart!"

	else:
		reply = "Don't know what to do with '" + msg + "'!"
	return reply


class SigHandler_guck:
	def __init__(self, mp_loggerqueue, mp_loglistener, state_data, logger):
		self.logger = logger
		self.state_data = state_data
		self.mp_loggerqueue = mp_loggerqueue
		self.mp_loglistener = mp_loglistener

	def sighandler_guck(self, a, b):
		global TERMINATED
		TERMINATED = True
		# self.shutdown(exit_status=1)

	def get_trstr(self, exit_status):
		if exit_status == 3:
			trstr = str(datetime.datetime.now()) + ": RESTART - "
		else:
			trstr = str(datetime.datetime.now()) + ": SHUTDOWN - "
		return trstr

	def shutdown(self, exit_status=1):
		trstr = self.get_trstr(exit_status)
		if self.state_data.TG and self.state_data.TG.running:
			self.state_data.TG.stop()
		if self.state_data.KB and self.state_data.KB.active:
			self.state_data.KB.stop()
			self.state_data.KB.join()
		mp_pd = self.state_data.mpp_peopledetection
		if mp_pd:
			if mp_pd.pid:
				print(trstr + "joining peopledetection ...")
				self.state_data.PD_OUTQUEUE.put("stop")
				mp_pd.join()
				print(self.get_trstr(exit_status) + "peopledetection exited!")
		mp_wf = self.state_data.mpp_webflask
		trstr = self.get_trstr(exit_status)
		if mp_wf:
			if mp_wf.pid:
				os.kill(mp_wf.pid, signal.SIGFPE)
				time.sleep(0.2)
				print(trstr + "joining flask webserver, this may take a while ...")
				os.kill(mp_wf.pid, signal.SIGTERM)
				mp_wf.join(timeout=10)
				if mp_wf.is_alive():
					os.kill(mp_wf.pid, signal.SIGQUIT)
					mp_wf.join()
				print(self.get_trstr(exit_status) + "flask webserver exited!")
		trstr = self.get_trstr(exit_status)
		if self.mp_loglistener:
			if self.mp_loglistener.pid:
				print(trstr + "joining loglistener ...")
				mplogging.stop_logging_listener(self.mp_loggerqueue, self.mp_loglistener)
				self.mp_loglistener.join(timeout=5)
				if self.mp_loglistener.is_alive():
					print(trstr + "killing loglistener")
					os.kill(self.mp_loglistener.pid, signal.SIGKILL)
				print(self.get_trstr(exit_status) + "loglistener exited!")


def input_raise_to(a, b):
	raise TimeoutError


def input_to(fn, timeout, queue):
	signal.signal(signal.SIGALRM, input_raise_to)
	signal.signal(signal.SIGINT, input_raise_to)
	signal.signal(signal.SIGTERM, input_raise_to)
	signal.alarm(timeout)
	sys.stdin = os.fdopen(fn)
	try:
		msg = input()
		signal.alarm(0)
		queue.put(msg)
	except TimeoutError:
		signal.alarm(0)
		queue.put(None)
	except Exception:
		pass


class StateData:
	def __init__(self):
		self.PD_ACTIVE = False
		self.mpp_peopledetection = None
		self.mpp_webflask = None
		self.TG = None
		self.KB = None
		self.PD_INQUEUE = None
		self.PD_OUTQUEUE = None
		self.NS_INQUEUE = None
		self.NS_OUTQUEUE = None
		self.MAINQUEUE = None
		self.DIRS = None
		self.DO_RECORD = False
		self.CAMERADATA = []
		self.CAMERA_RESTART_TIME = {}
		self.CAMERA_CONFIG = []
		self.SSH_CONFIG = []



class KeyboardThread(Thread):
	def __init__(self, state_data, cfgr, mp_loggerqueue, logger):
		Thread.__init__(self)
		self.daemon = True
		self.state_data = state_data
		self.mp_loggerqueue = mp_loggerqueue
		self.pd_inqueue = self.state_data.PD_INQUEUE
		self.pd_outqueue = self.state_data.PD_OUTQUEUE
		self.cfgr = cfgr
		self.logger = logger
		self.running = False
		self.active = self.get_config()
		self.kbqueue = mp.Queue()
		self.fn = sys.stdin.fileno()
		self.is_shutdown = False

	def get_config(self):
		active = self.cfgr.get_options()["keyboard_active"]
		return active

	def sighandler_kbd(self, a, b):
		self.running = False

	def send_message_all(self, txt):
		if not self.active:
			return
		print(txt)

	def send_photo(self, photopath, caption):
		pass

	def stop(self):
		if not self.active:
			return
		self.running = False
		self.logger.debug("stopping keyboard thread")
		print("Stopping " +  __appname__ + " keyboard bot, this may take a second ...")

	def run(self):
		if not self.active:
			return
		self.logger.debug("starting keyboard thread")
		self.running = True
		instruction = ">> Enter '?' or 'help' for help"
		print(instruction)
		while self.running:
			mpp_inputto = mp.Process(target=input_to, args=(self.fn, 1, self.kbqueue,))
			mpp_inputto.start()
			msg = self.kbqueue.get()
			mpp_inputto.join()
			if self.running and msg:
				reply = GeneralMsgHandler(msg, "kbd", self.state_data)
				print(reply)
				print(instruction)
		self.logger.debug("keyboard thread stopped!")


class TelegramThread(Thread):
	def __init__(self, state_data, cfgr, mp_loggerqueue, logger):
		Thread.__init__(self)
		self.state_data = state_data
		self.mp_loggerqueue = mp_loggerqueue
		self.pd_inqueue = self.state_data.PD_INQUEUE
		self.pd_outqueue = self.state_data.PD_OUTQUEUE
		self.cfgr = cfgr
		self.logger = logger
		self.active, self.token, self.chatids = self.get_config()
		self.running = False
		self.heartbeatok = True

	def run(self):
		if not self.active:
			return -1

		self.logger.debug("starting telegram handler")
		self.logger.debug("telegram  token & chat ids: " + str(self.token) + " / " + str(self.chatids))

		self.running = True

		rep = "Welcome to "+  __appname__ + " / " + __version__ + "! Enter '?' or 'help' for help\n"
		fg.send_message(self.token, self.chatids, rep)
		lastt0 = time.time()
		last_tg_cleanup = time.time()
		clearbot_answer = self.clear_bot(self.token)
		self.logger.info("Received answer on first clear_bot: " + str(clearbot_answer))
		self.running = True
		self.heartbeatok = True
		self.logger.info("telegram handler/bot started!")
		while self.running:
			try:
				# self.logger.debug("starting receive_message")
				# ok, rlist, err0 = self.receive_message()
				ok, rlist, err0 = fg.receive_message(self.token)
				if ok and rlist:
					lastt0 = time.time()
					for chat_id, text in rlist:
						text = str(text)
						self.logger.info("Received message >" + text + "<")
						reply = GeneralMsgHandler(text.lower(), "tgram", self.state_data)
						fg.send_message(self.token, [chat_id], reply)
						if reply.startswith("collecting photo snapshots"):
							imglist = get_free_photos(self.state_data.DIRS["photo"],
													  self.state_data.CAMERA_CONFIG, self.logger)
							if imglist:
								for photo_path in imglist:
									# file_opened = open(photo_name, "rb")
									photo_name = os.path.basename(photo_path)
									fg.send_filepath_as_photo(self.token, [chat_id], photo_path, photo_name)
				elif time.time() - last_tg_cleanup > _CLEAR_BOT_FREQ * 60 * 60:  # every 6h
					self.logger.info("clearing telegram bot chat ...")
					clearbot_answer = fg.clear_bot(self.token)
					self.logger.info("Received answer on clear_bot: " + str(clearbot_answer))
					last_tg_cleanup = time.time()
				elif time.time() - lastt0 > _HEARTBEAT_FRQ * 60 or not ok:
					self.logger.debug("Sending getme - heartbeat to bot ...")
					heartbeat_answer = fg.get_me(self.token)
					if not heartbeat_answer:
						self.logger.warning(
							"Received no answer on getme, trying again ..."
						)
					# self.running = False
					# rep = "Shutting down Dachshund on missing getme - heartbeat ..."
					else:
						self.logger.info(
							"Received answer on getme: " + str(heartbeat_answer)
						)
					lastt0 = time.time()
				time.sleep(1)
			except Exception as e:
				self.logger.error("Telegram Thread Error: " + str(e) + ", clearing bot ...")
				clearbot_answer = fg.clear_bot(self.token)
				self.logger.info("Received answer on clear_bot: " + str(clearbot_answer))
				last_tg_cleanup = time.time()
		self.logger.warning("Exiting telegram thread")
		return 1

	def get_updates(self, token, offset=0):
		try:
			if int(offset) != 0:
				urlstr = (
					f"https://api.telegram.org/bot{token}/getUpdates?offset={str(offset)}"
				)
				answer = requests.get(urlstr)
				self.logger.info("*** Got get_updates answer: " + str(answer.content))
			else:
				urlstr = f"https://api.telegram.org/bot{token}/getUpdates"
				answer = requests.get(f"https://api.telegram.org/bot{token}/getUpdates")
		except Exception as e:
			return {"ok": False, "result": []}
		return json.loads(answer.content)


	def clear_bot(self, token):
		r = self.get_updates(token, offset=-1)
		if not r["ok"]:
			return False
		if not r["result"]:
			# if results empty but ok, return True / cleared!
			return True
		try:
			rlast = int(r["result"][-1]["update_id"] + 1)
			r = self.get_updates(token, offset=int(rlast))
			return r
		except Exception:
			return False

	def stop(self):
		if not self.active or not self.running:
			return
		self.logger.debug("stopping telegram bot")
		rep = "Stopping  telegram bot, this may take a while ..."
		fg.send_message(self.token, self.chatids, rep)
		self.logger.info("telegram bot stopped!")
		self.running = False

	def send_message_all(self, txt):
		if not self.active:
			return
		fg.send_message(self.token, self.chatids, txt)

	def send_photo(self, photo_path, caption):
		if not self.active:
			return
		fg.send_filepath_as_photo(self.token, self.chatids, photo_path, caption)

	def get_config(self):
		t = self.cfgr.get_telegram()
		if t["active"].lower() == "no":
			return False, None, None
		try:
			token = t["token"]
			chatids = json.loads(t["chatids"])
			self.logger.debug("got config for active telegram bot")
		except Exception as e:
			self.logger.debug(str(e) + "telegram config error, setting telegram to inactive!")
			return False, None, None
		return True, token, chatids


def mainloop():

	global TERMINATED
	global RESTART

	TERMINATED = False
	RESTART = False

	setproctitle(__appabbr__ + "." + os.path.basename(__file__))

	# get dirs
	ret, dirs = setup_dirs(__version__)

	if __startmode__ == "systemd":
		printlogdir = dirs["logs"] + "printlog.txt"
		try:
			sys.stdout = open(printlogdir, "w")
		except Exception:
			pass
	print("*" * 80)
	print(str(datetime.datetime.now()) + ": START UP - starting GUCK " + __version__)
	if psutil.Process(os.getpid()).ppid() == 1:
		print(" ... We are using systemd")
	else:
		print(" ... We are NOT using systemd!")
	if ret == -1:
		print(dirs)
		print(str(datetime.datetime.now()) + ": START UP - " + dirs)
		print(str(datetime.datetime.now()) + ": START UP - exiting ...")
	else:
		print(str(datetime.datetime.now()) + ": START UP - setup for folders ok!")

	# read GUCK config
	try:
		cfg_file = dirs["configfile"]
		cfg = configparser.ConfigParser()
		cfg.read(cfg_file)
	except Exception as e:
		print(str(datetime.datetime.now()) + ": START UP - " + str(e) + ": config file syntax error, exiting")
		return -1

	# get log level
	try:
		loglevel_str = cfg["OPTIONS"]["LOGLEVEL"].lower()
		if loglevel_str == "info":
			loglevel = logging.INFO
		elif loglevel_str == "debug":
			loglevel = logging.DEBUG
		elif loglevel_str == "warning":
			loglevel = logging.WARNING
		elif loglevel_str == "error":
			loglevel = logging.ERROR
		else:
			loglevel = logging.INFO
			loglevel_str = "info"
	except Exception:
		loglevel = logging.INFO
		loglevel_str = "info"
	print(str(datetime.datetime.now()) + ": START UP - setting log level to " + loglevel_str)

	print(str(datetime.datetime.now()) + ": START UP - now switching to logging in log files!")

	# global data object
	state_data = StateData()
	state_data.DIRS = dirs

	# get camera data
	cfgr = ConfigReader(cfg)
	state_data.CAMERA_CONFIG = cfgr.get_cameras()

	# get ssh data & commands
	state_data.SSH_CONFIG = cfgr.get_ssh()

	# init logger
	print(str(datetime.datetime.now()) + ": main log file is - " + dirs["logs"] + "guck.log")
	mp_loggerqueue, mp_loglistener = mplogging.start_logging_listener(dirs["logs"] + "guck.log", maxlevel=loglevel)
	logger = mplogging.setup_logger(mp_loggerqueue, __file__)
	logger.debug("starting with loglevel '" + loglevel_str + "'")
	logger.info("Welcome to GUCK " + __version__)
	logger.info("started with startmode " + __startmode__)

	# sighandler
	sh = SigHandler_guck(mp_loggerqueue, mp_loglistener, state_data, logger)
	old_sigint = signal.getsignal(signal.SIGINT)
	old_sigterm = signal.getsignal(signal.SIGTERM)
	signal.signal(signal.SIGINT, sh.sighandler_guck)
	signal.signal(signal.SIGTERM, sh.sighandler_guck)

	# save photos setup
	try:
		options = cfgr.get_options()
		addtl_photo_path = options["addtl_photo_path"]
		if addtl_photo_path.lower() == "none":
			addtl_photo_path = None
	except Exception:
		addtl_photo_path = None

	# init queues
	state_data.PD_INQUEUE = mp.Queue()
	state_data.PD_OUTQUEUE = mp.Queue()
	state_data.MAINQUEUE = queue.Queue()
	state_data.WF_INQUEUE = mp.Queue()
	state_data.WF_OUTQUEUE = mp.Queue()
	state_data.NS_INQUEUE = mp.Queue()
	state_data.NS_OUTQUEUE = mp.Queue()

	# WebServer
	try:
		webflask.REDISCLIENT.ping()
	except Exception as e:
		logger.error(str(e) + ": cannot start webserver due to redis server not available, exiting")
		sh.shutdown()
		return -1
	state_data.mpp_webflask = mp.Process(target=webflask.main, args=(cfg, dirs, state_data.WF_OUTQUEUE,
																	 state_data.WF_INQUEUE, mp_loggerqueue,))
	state_data.mpp_webflask.start()
	if state_data.WF_INQUEUE.get() == "False":
		logger.error(": cannot init DB, exiting")
		sh.shutdown()
		return -1

	commlist = []
	# Telegram
	logger.info("starting telegram ...")
	state_data.TG = TelegramThread(state_data, cfgr, mp_loggerqueue, logger)
	state_data.TG.start()
	commlist.append(state_data.TG)

	# KeyboardThread
	if __startmode__ != "systemd":
		state_data.KB = KeyboardThread(state_data, cfgr, mp_loggerqueue, logger)
		state_data.KB.start()
		commlist.append(state_data.KB)
	else:
		state_data.KB = None

	wf_msglist = []
	pd_cmd = None

	lastt_stdout = time.time()
	lastt_camcheck = time.time() - 4*60

	while not TERMINATED:

		time.sleep(0.01)
		if time.time() - lastt_stdout > 5:
			sys.stdout.flush()
			lastt_stdout = time.time()

		# get from webflask queue
		try:
			wf_cmd, wf_data = state_data.WF_INQUEUE.get_nowait()
			if wf_cmd == "get_pd_status":
				state_data.WF_OUTQUEUE.put((state_data.PD_ACTIVE, wf_msglist))
				wf_msglist = []
			elif wf_cmd == "set_pdstart":
				state_data.WF_OUTQUEUE.put((state_data.PD_ACTIVE, None))
				pd_cmd = "start"
			elif wf_cmd == "set_pdstop":
				state_data.WF_OUTQUEUE.put((state_data.PD_ACTIVE, None))
				pd_cmd = "stop"
			elif wf_cmd == "set_pdrestart":
				state_data.WF_OUTQUEUE.put((state_data.PD_ACTIVE, None))
				pd_cmd = "restart!!"
			elif wf_cmd == "get_host_status":
				ret, mem_crit, cpu_crit, gpu_crit, cam_crit = get_status(state_data, __version__)
				state_data.WF_OUTQUEUE.put(("status", (ret, mem_crit, cpu_crit, gpu_crit, cam_crit)))
			elif wf_cmd == "get_free_photodata":
				imglist = get_free_photos(dirs["photo"], state_data.CAMERA_CONFIG, logger)
				state_data.WF_OUTQUEUE.put(("free_photodata", imglist))
		except (queue.Empty, EOFError):
			pass
		except Exception:
			pass

		# get el from peopledetection queue (clear it always!!)
		pdmsglist = []
		while True:
			try:
				pdmsglist = state_data.PD_INQUEUE.get_nowait()
			except (queue.Empty, EOFError):
				break
			except Exception:
				break

		if pdmsglist:
			state_data.CAMERADATA = []
			for pdmsg, pdpar in pdmsglist:
				c_cname, c_frame, _, _, _, _ = pdpar
				state_data.CAMERADATA.append(pdpar)
				if pdmsg == "detection":
					try:
						logger.info("received detection for " + c_cname)
						datestr = datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S")
						photo_caption = datestr + ": Object detected @ " + c_cname + "!"
						short_photo_name = c_cname + "_" + datestr + ".jpg"
						photo_name = dirs["photo"] + short_photo_name
						wf_msglist.insert(0, photo_name)
						# save it to photo_dir
						cv2.imwrite(photo_name, c_frame)
						# send photo
						for c in commlist:
							c.send_photo(photo_name, photo_caption)
						if addtl_photo_path:
							photo_name2 = addtl_photo_path + c_cname + "_" + datestr + ".jpg"
							cv2.imwrite(photo_name2, c_frame)
					except Exception as e:
						logger.warning(str(e))

		# get el from main queue (GeneralMsgHandler)
		# because we cannot start pdedector from thread! (keras/tf bug/feature!?)
		try:
			mq_cmd = None
			mq_param = None
			if not pd_cmd:
				mq_cmd, mq_param = state_data.MAINQUEUE.get_nowait()
				print(mq_cmd)
			else:
				if pd_cmd == "start":
					mq_cmd = "start"
					# mq_param = "wf"
				elif pd_cmd == "stop":
					mq_cmd = "stop"
					mq_param = "wf"
				elif pd_cmd == "restart!!":
					mq_cmd = "restart!!"
					mq_param = "wf"
				elif pd_cmd == "camrestart":
					mq_cmd = "camrestart"
					mq_param = "wf"
				pd_cmd = None
			if mq_cmd == "start" and not state_data.PD_ACTIVE:
				mpp_peopledetection = mp.Process(target=peopledetection.run_cameras,
												 args=(state_data.PD_INQUEUE, state_data.PD_OUTQUEUE, state_data.DIRS,
													   mq_param, cfg, mp_loggerqueue,))
				mpp_peopledetection.start()
				state_data.mpp_peopledetection = mpp_peopledetection
				state_data.PD_OUTQUEUE.put((mq_param + "_active", True))
				try:
					pd_answer, pd_prm = state_data.PD_INQUEUE.get()
					if "error" in pd_answer:
						state_data.PD_ACTIVE = False
						logger.error(": cameras/PD startup failed!")
						state_data.mpp_peopledetection.join()
						for c in commlist:
							c.send_message_all("Error - cannot start GUCK people detection! (" +
											   str(pd_answer) + ")")
					else:
						logger.info("cameras/PD started!")
						state_data.PD_ACTIVE = True
						for c in commlist:
							c.send_message_all("... GUCK people detection started! (" +
											   str(pd_prm) + ")")
							
				except Exception as e:
					logger.error(str(e) + ": cannot communicate with peopledetection, trying to exit!")
					state_data.PD_ACTIVE = False
					try:
						os.kill(mpp_peopledetection.pid, signal.SIGKILL)
						mpp_peopledetection.join(timeout=5)
					except Exception:
						pass
					TERMINATED = True
			elif mq_cmd == "stop":
				if state_data.mpp_peopledetection:
					if state_data.mpp_peopledetection.pid:
						state_data.PD_OUTQUEUE.put(("stop", None))
						state_data.mpp_peopledetection.join(timeout=5)
						if state_data.mpp_peopledetection.is_alive():
							os.kill(state_data.mpp_peopledetection.pid, signal.SIGTERM)
						state_data.mpp_peopledetection = None
						for c in commlist:
							c.send_message_all("... GUCK people detection stopped!")
					state_data.PD_ACTIVE = False
			elif mq_cmd == "exit!!" or mq_cmd == "restart!!":
				if mq_cmd == "restart!!":
					RESTART = True
				if state_data.mpp_peopledetection:
					if state_data.mpp_peopledetection.pid:
						state_data.PD_OUTQUEUE.put(("stop", None))
						state_data.mpp_peopledetection.join()
						state_data.PD_ACTIVE = False
				TERMINATED = True
			elif mq_cmd == "camrestart":
				camnr = mq_param

		except (queue.Empty, EOFError):
			pass
		except Exception:
			pass

	# shutdown
	exitcode = 1
	if RESTART:
		exitcode = 3
	# close all the other mps & stuff
	logger.info("calling shutdown sequence ...")
	sh.shutdown(exitcode)
	print(str(datetime.datetime.now()) + "... shutdown sequence finished!")

	#signal.signal(signal.SIGINT, old_sigint)
	#signal.signal(signal.SIGTERM, old_sigterm)
	clear_all_queues([state_data.PD_INQUEUE, state_data.PD_OUTQUEUE, state_data.MAINQUEUE])
	return exitcode

