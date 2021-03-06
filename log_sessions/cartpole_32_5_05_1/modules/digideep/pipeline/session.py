import os, sys, argparse
import fcntl
import re
from shutil import copytree, copyfile, ignore_patterns
import pipes
# import subprocess
# import threading
# import time

from digideep.utility.logging import logger
from digideep.utility.toolbox import dump_dict_as_json, dump_dict_as_yaml, get_module
from digideep.utility.json_encoder import JsonDecoder
from digideep.utility.monitoring import monitor
from digideep.utility.profiling import profiler, KeepTime
from digideep.utility.name_generator import make_unique_path_session

import pickle, torch
from copy import deepcopy


def print_verbose(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


def check_session(path, verbose=False):
    print_verbose("Checking if provided path is a valid session.", verbose=verbose)
    print_verbose("path =", path, verbose=verbose)

    print_verbose("  Checking if a directory ...", end="", verbose=verbose)
    if not os.path.isdir(path):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if 'checkpoints' directory exists ...", end="", verbose=verbose)
    if not os.path.isdir(os.path.join(path, "checkpoints")):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if 'modules' directory exists ...", end="", verbose=verbose)
    if not os.path.isdir(os.path.join(path, "modules")):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if '__init__.py' file exists ...", end="", verbose=verbose)
    if not os.path.isfile(os.path.join(path, "__init__.py")):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if 'params.yaml' file exists ...", end="", verbose=verbose)
    if not os.path.isfile(os.path.join(path, "params.yaml")):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if 'cpanel.json' file exists ...", end="", verbose=verbose)
    if not os.path.isfile(os.path.join(path, "cpanel.json")):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if 'saam.py' file exists ...", end="", verbose=verbose)
    if not os.path.isfile(os.path.join(path, "saam.py")):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("Provided path is a valid session.", verbose=verbose)
    print_verbose("", verbose=verbose)
    return True


def check_checkpoint(path, verbose=False):
    print_verbose("Checking if provided path is a valid checkpoint.", verbose=verbose)
    print_verbose("path =", path, verbose=verbose)

    print_verbose("  Checking if a directory ...", end="", verbose=verbose)
    if not os.path.isdir(path):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking name pattern to match 'checkpoint-\d+' ...", end="", verbose=verbose)
    if not bool(re.match("checkpoint-\d+", os.path.split(path)[1])):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if runner.pt exists ...", end="", verbose=verbose)
    if not os.path.isfile(os.path.join(path, "runner.pt")):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("  Checking if ../.. is a session ...", end="", verbose=verbose)
    if not check_session(os.path.dirname(os.path.dirname(path)), verbose=False):
        print_verbose("   NO", verbose=verbose)
        return False
    print_verbose("   YES", verbose=verbose)

    print_verbose("Provided path is a valid checkpoint.", verbose=verbose)
    print_verbose("", verbose=verbose)
    return True


# class ParCommand(threading.Thread):
#     def __init__(self, command, logger):
#         self.stdout = None
#         self.stderr = None
#         self.command = command
#         self.logger = logger
#         threading.Thread.__init__(self)
#     def run(self):
#         t = time.time()
#         p = subprocess.Popen(self.command,
#                              shell=False,
#                              stdout=subprocess.PIPE,
#                              stderr=subprocess.PIPE)
#         self.stdout, self.stderr = p.communicate()
#         p.wait()
#         # self.logger.warn(" Thread '", " ".join(self.command), "' is over with exit code = ", p.returncode, " in ",time.time()-t," seconds.", sep="")
#         self.logger.warn("Command: '{}' is over with exit code: {} in {:6.2f} seconds".format(" ".join(self.command),
#                                                                                              p.returncode,
#                                                                                              time.time()-t))
#
#         # t.wait()
#         # t.poll()


writers = []


class Session(object):
    """
    This class provides the utilities for storing results of a session.
    It provides a unique path based on a timestamp and creates all sub-
    folders that are required there. A session directory will have the
    following contents:
    * :file:`session_YYYYMMDDHHMMSS/`:
        * :file:`checkpoints/`: The directory of all stored checkpoints.
        * :file:`modules/`: A copy of all modules that should be saved with the results. This helps to load
          checkpoints in evolving codes with breaking changes. Use extra modules with ``--save-modules``
          command-line option.
        * :file:`monitor/`: Summary results of each worker environment.
        * :file:`cpanel.json`: A json file including control panel (``cpanel``) parameters in ``params`` file.
        * :file:`params.yaml`: The parameter tree of the session, i.e. the params variable in ``params`` file.
        * :file:`report.log`: A log file for Logger class.
        * :file:`visdom.log`: A log file for visdom logs.
        * :file:`__init__.py`: Python ``__init__`` file to convert the session to a module.
    .. comment out this part
    .. * :file:`loader.py`: A helping module for loading saved checkpoints more intuitively.
    Arguments:
        root_path (str): The path to the ``digideep`` module.
    Note:
        This class also initializes helping tools (e.g. Visdom, Logger, Monitor,
        etc.) and has helper functions for saving/loading checkpoints.
    Tip:
        The default directory for storing sessions is :file:`/tmp/digideep_sessions`.
        To change the default directory use the program with cli argument ``--session-path <path>``
    Todo:
      Complete the session-as-a-module (SaaM) implementation. Then, :file:`session_YYYYMMDDHHMMSS`
      should work like an importable module for testing and inference.
    Todo:
      If restoring a session, ``visdom.log`` should be copied from there and replayed.
                                         play    resume    loading    dry-run    session-only    |  implemented
    -------------------------------------------------------------------------------------------- | ------------
    Train                                 0         0         0          0            0          |      1
    Train session barebone                0         0         0          0            1          |      1
    Train from a checkpoint               0         1         1          0            0          |      1
    Play (policy initialized)             1         0         0         0/1           0          |      1
    Play (policy loaded from checkpoint)  1         0         1         0/1           0          |      1
    """

    def __init__(self, root_path):
        self.parse_arguments()
        self.state = {}
        # If '--dry-run' is specified no reports should be generated. It is not relevant to whether
        # we are loading from a checkpoint or running from scratch. If dry-run is there no reports
        # should be generated.
        self.dry_run = True if self.args["dry_run"] else False

        self.is_loading = True if self.args["load_checkpoint"] else False
        self.is_playing = True if self.args["play"] else False
        self.is_resumed = True if self.args["resume"] else False
        self.is_customs = True if self.args["custom"] else False
        self.is_session_only = True if self.args["create_session_only"] else False

        assert (self.is_loading and self.is_playing) or (self.is_loading and self.is_resumed) or (
                    self.is_loading and self.is_customs) or (not self.is_loading), \
            "--load-checkpoint argument should be used either with --play, --resume, or --custom arguments."
        assert (self.is_session_only and (not self.is_loading) and (not self.is_playing) and (not self.is_resumed) and (
            not self.is_customs)) or (not self.is_session_only), \
            "--create-session-only argument cannot be used with any of the --load-checkpoint, --play, --resume, or --custom arguments."

        # Automatically find the latest checkpoint if not specified
        self.state['checkpoint_name'] = None
        if self.is_loading:
            if check_checkpoint(self.args["load_checkpoint"], verbose=True):
                self.state['checkpoint_name'] = os.path.split(self.args["load_checkpoint"])[1]
            elif check_session(self.args["load_checkpoint"], verbose=True):
                last_checkpoint = sorted([int(d.replace("checkpoint-", "")) for d in
                                          os.listdir(os.path.join(self.args["load_checkpoint"], "checkpoints"))])[-1]
                self.args["load_checkpoint"] = os.path.join(self.args["load_checkpoint"], "checkpoints",
                                                            "checkpoint-" + str(last_checkpoint))
                self.state['checkpoint_name'] = "checkpoint-" + str(last_checkpoint)
            else:
                raise ValueError("In '--load-checkpoint path', path is neither a valid checkpoint nor a valid session.")

        # TODO: Change the path for loading the packages?
        # sys.path.insert(0, '/path/to/whatever')

        # if self.args["monitor_cpu"] or self.args["monitor_gpu"]:
        #     # Force visdom ON if "--monitor-cpu" or "--monitor-gpu" are provided.
        #     self.args["visdom"] = True

        # Root: Indicates where we are right now
        self.state['path_root'] = os.path.split(root_path)[0]

        # Session: Indicates where we want our codes to be stored
        if self.is_loading and self.is_playing:
            # If we are playing a recorded checkpoint, we must save the results into the `evaluations` path
            # of that session.
            checkpoint_path = os.path.split(self.args["load_checkpoint"])[0]
            self.state['path_base_sessions'] = os.path.join(os.path.split(checkpoint_path)[0], "evaluations")
        elif self.is_loading and self.is_resumed:
            if self.args['session_name']:
                print("Warning: --session-name is ignored.")

            directory = os.path.dirname(os.path.dirname(self.args["load_checkpoint"]))
            self.state['path_base_sessions'] = os.path.split(directory)[0]
            self.args['session_name'] = os.path.split(directory)[1]
        elif self.is_loading and self.is_customs:
            # If we are doing a custom task from a checkpoint, we must save the results into the `customs` path
            # of that session.
            checkpoint_path = os.path.split(self.args["load_checkpoint"])[0]
            self.state['path_base_sessions'] = os.path.join(os.path.split(checkpoint_path)[0], "customs")
        else:
            # OK, we are loading from a checkpoint, just create session from scratch.
            # self.state['path_root_session']  = self.args["session_path"]
            # self.state['path_base_sessions'] = os.path.join(self.state['path_root_session'], 'digideep_sessions')
            self.state['path_base_sessions'] = self.args["session_path"]

        # 1. Creating 'path_base_sessions', i.e. '/tmp/digideep_sessions':
        try:  # TODO: and not self.dry_run:
            os.makedirs(self.state['path_base_sessions'])
            # Create an empty __init__.py in it!
        except FileExistsError:
            pass
        except Exception as ex:
            print(ex)

        try:
            with open(os.path.join(self.state['path_base_sessions'], '__init__.py'), 'w') as f:
                print("", file=f)
        except FileExistsError:
            pass
        except Exception as ex:
            print(ex)

        # 2. Create a unique 'path_session':
        if not self.dry_run:
            if self.args['session_name']:
                # If is_loading then this line will be executed ...
                self.state['path_session'] = os.path.join(self.state['path_base_sessions'], self.args["session_name"])
                # TODO: Make the directory
                try:
                    os.makedirs(self.state['path_session'])
                except Exception as ex:
                    print(ex)
            else:
                self.state['path_session'] = make_unique_path_session(self.state['path_base_sessions'],
                                                                      prefix="session_")
        else:
            self.state['path_session'] = os.path.join(self.state['path_base_sessions'], "no_session")

        # This will be equal to args['session_name'] if that has existed previously.
        self.state['session_name'] = os.path.split(self.state['path_session'])[-1]

        self.state['path_checkpoints'] = os.path.join(self.state['path_session'], 'checkpoints')
        self.state['path_memsnapshot'] = os.path.join(self.state['path_session'], 'memsnapshot')
        self.state['path_monitor'] = os.path.join(self.state['path_session'], 'monitor')
        self.state['path_videos'] = os.path.join(self.state['path_session'], 'videos')
        self.state['path_tensorboard'] = os.path.join(self.state['path_session'], 'tensorboard')
        # Hyper-parameters basically is a snapshot of intial parameter engine's state.
        self.state['file_cpanel'] = os.path.join(self.state['path_session'], 'cpanel.json')
        self.state['file_repeal'] = os.path.join(self.state['path_session'], 'repeal.json')
        self.state['file_params'] = os.path.join(self.state['path_session'], 'params.yaml')

        self.state['file_report'] = os.path.join(self.state['path_session'], 'report.log')
        # self.state['file_visdom'] = os.path.join(self.state['path_session'], 'visdom.log')
        self.state['file_varlog'] = os.path.join(self.state['path_session'], 'varlog.json')
        self.state['file_prolog'] = os.path.join(self.state['path_session'], 'prolog.json')
        self.state['file_monlog'] = os.path.join(self.state['path_session'], 'monlog.json')
        self.state['lock_running'] = os.path.join(self.state['path_session'], 'running.lock')
        self.state['lock_done'] = os.path.join(self.state['path_session'], 'done.lock')

        # Here, the session path has been created or it existed.
        # Now make sure only one instance passes from this point.
        self.check_singleton_instance()
        self.check_if_done()

        # 3. Creating the rest of paths:
        if not self.is_playing and not self.is_resumed and not self.dry_run:
            os.makedirs(self.state['path_checkpoints'])
            os.makedirs(self.state['path_memsnapshot'])
        if not self.is_resumed and not self.dry_run:
            os.makedirs(self.state['path_monitor'])

        self.initLogger()
        self.initVarlog()
        self.initProlog()
        self.initTensorboard()
        # self.initVisdom()
        # TODO: We don't need the "SaaM" when are loading from a checkpoint.
        # if not self.is_playing:
        self.createSaaM()
        #################
        self.runMonitor()  # Monitor CPU/GPU/RAM
        self.set_device()

        # Check valid params file:
        if not self.is_loading:
            try:
                get_module(self.args["params"])
            except Exception as ex:
                logger.fatal("While importing user-specified params:", ex)
                exit()
        if self.is_loading:
            logger.warn("Loading from:", self.args["load_checkpoint"])

        if not self.dry_run:
            print(':: The session will be stored in ' + self.state['path_session'])
        else:
            print(':: This session has no footprints. Use without `--dry-run` to store results.')

    def initTensorboard(self):
        """
        Will initialize the SummaryWriter for tensorboard logging.
        Link: https://pytorch.org/docs/stable/tensorboard.html
        """
        # TODO: Is it required?

        from torch.utils.tensorboard import SummaryWriter
        # Todo : Automatic
        self.writer = SummaryWriter(log_dir='logs/cartpole_32_5_05_1')

        # Put it here for global access to tensorboard!
        writers.append(self.writer)

        if self.args["tensorboard"]:
            # Run a dedicated Tensorboard server:
            from tensorboard import program
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--bind_all', '--logdir', self.state['path_tensorboard']])
            url = tb.launch()
            logger.warn("Access Tensorboard through: " + str(url))
        else:
            # Nullify the attributes so time would not be wasted logging.
            for attr in dir(self.writer):
                if attr.startswith("add_") or (attr == "flush") or (attr == "close"):
                    setattr(self.writer, attr, lambda *args, **kw: None)

    def save_states(self, states, index):
        if self.dry_run:
            return
        import torch
        dirname = os.path.join(self.state['path_checkpoints'], "checkpoint-"+'0')
        # Todo : Automatic
        #dirname = 'saved_models_checkpoints/picainmodel_128_16'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, "states.pt")
        torch.save(states, filename)

    def save_runner(self, runner, index):
        if self.dry_run:
            return
        dirname = os.path.join(self.state['path_checkpoints'], "checkpoint-" + '0')
        #dirname = 'saved_models_checkpoints/picainmodel_128_16'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, "runner.pt")
        pickle.dump(runner, open(filename, "wb"), pickle.HIGHEST_PROTOCOL)
        print(">> 5")
        logger.warn('>>> Network runner saved at {}\n'.format(dirname))



    def finalize(self):
        logger.fatal("\n", '=' * 50, "\n", "\n" * 5, " " * 15, "END OF SIMULATION\n", "\n" * 5, "=" * 50, "\n" * 5,
                     sep="")

    def initLogger(self):
        """
        This function sets the logger level and file.
        """
        if not self.dry_run:
            logger.set_logfile(self.state['file_report'])
        logger.set_log_level(self.args["log_level"])

    def initVarlog(self):
        if not self.dry_run:
            monitor.set_output_file(self.state['file_varlog'])

    def initProlog(self):
        if not self.dry_run:
            profiler.set_output_file(self.state['file_prolog'])
        KeepTime.set_level(self.args["profiler_level"])

    # def initVisdom(self):
    #     """
    #     This function initializes the connection to the Visdom server. The Visdom server must be running.
    #
    #     .. code-block:: bash
    #         :caption: Running visdom server
    #
    #         visdom -port 8097 &
    #     """
    #     if self.args["visdom"]:
    #         from digideep.utility.visdom_engine.Instance  import VisdomInstance
    #         if not self.dry_run:
    #             VisdomInstance(port=self.args["visdom_port"], log_to_filename=self.state["file_visdom"], replay=True)
    #         else:
    #             VisdomInstance(port=self.args["visdom_port"])

    def createSaaM(self):
        """ SaaM = Session-as-a-Module
        This function will make the session act like a python module.
        The user can then simply import the module for inference.
        """
        if self.dry_run or self.is_loading:
            return
        # Copy the all modules
        modules = set(self.args["save_modules"])
        # Add digideep per se to the saved modules.
        modules.add("digideep")
        modules_path = os.path.join(self.state['path_session'], 'modules')
        for mod in modules:
            real_mod = get_module(mod)
            module_source = real_mod.__path__[0]
            module_target = os.path.join(modules_path, mod)
            copytree(module_source, module_target, ignore=ignore_patterns('*.pyc', '__pycache__'))
            if mod == "digideep":
                digideep_path = module_source

        # Copy saam.py to the session root path
        copyfile(os.path.join(digideep_path, 'saam.py'), os.path.join(self.state['path_session'], 'saam.py'))
        # Create __init__.py at the session root path
        with open(os.path.join(self.state['path_session'], '__init__.py'), 'w') as f:
            print("from .saam import loader", file=f)
            # print("from .loader import ModelCarousel", file=f)

    def runMonitor(self):
        """
        This function will load the monitoring tool for CPU and GPU utilization and memory consumption.
        """
        if (not self.args["no_monitor_cpu"]) or (not self.args["no_monitor_gpu"]):
            from digideep.utility.stats import StatLogger
            st = StatLogger(monitor_cpu=not self.args["no_monitor_cpu"],
                            monitor_gpu=not self.args["no_monitor_gpu"],
                            output=self.state['file_monlog'])
            # interval=self.args["monitor_interval"], window=self.args["monitor_window"])
            st.start()

    def update_params(self, params):
        params['session_name'] = self.state['session_name']
        params['session_msg'] = self.args['msg']
        params['session_cmd'] = 'python ' + ' '.join(pipes.quote(x) for x in sys.argv)
        return params

    def dump_cpanel(self, cpanel):
        if self.dry_run:
            return
        dump_dict_as_json(self.state['file_cpanel'], cpanel)

    def dump_repeal(self, repeal):
        if self.dry_run:
            return
        dump_dict_as_json(self.state['file_repeal'], repeal)

    def dump_params(self, params):
        if self.dry_run:
            return
        dump_dict_as_yaml(self.state['file_params'], params)

    def set_device(self):
        ## CPU
        # Sets the number of OpenMP threads used for parallelizing CPU operations
        torch.set_num_threads(1)

        ## GPU
        cuda_available = torch.cuda.is_available()
        if cuda_available:  # and use_gpu:
            logger("GPU available. Using 1 GPU.")
            self.device = torch.device("cuda")
            # self.dtype = torch.cuda.FloatTensor
            # self.dtypelong = torch.cuda.LongTensor
        else:
            logger("Using CPUs.")
            self.device = torch.device("cpu")
            # self.dtype = torch.FloatTensor
            # self.dtypelong = torch.LongTensor

        # TODO: For debugging
        # self.device = torch.device("cpu")

    def get_device(self):
        return self.device

    # def create_running_lock(self):
    # with open(self.state['lock_running'], 'w') as f:
    #     print("", file=f)

    ####################################
    ## Locks: done.lock, running.lock ##
    ####################################
    def check_singleton_instance(self):
        if self.is_playing or self.is_customs:
            return
        if not self.is_loading:
            # Create running.lock for the first time
            with open(self.state['lock_running'], 'w') as f:
                print("", file=f)

        lock_file_pointer = os.open(self.state['lock_running'], os.O_WRONLY)
        try:
            fcntl.lockf(lock_file_pointer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            print("ERROR: Another instance is already running. Make sure to exit from the other one to continue.",
                  file=sys.stderr)
            print("       Currently running {}".format(self.state['path_session']), file=sys.stderr)
            sys.exit(2)

    def check_if_done(self):
        if self.is_playing or self.is_customs:
            return
        if not self.is_loading:
            return
        if os.path.isfile(self.state['lock_done']):
            print("ERROR: This instance is already done.", file=sys.stderr)
            print("       Currently running {}".format(self.state['path_session']), file=sys.stderr)
            sys.exit(3)

    def mark_as_done(self):
        # os.path.join(self.state['path_session'], 'done.lock')
        with open(self.state['lock_done'], 'w') as f:
            print("", file=f)

    # def task_rsync(self, source, target):
    #     # -a: Archive
    #     # -z: Compress (makes it very slow)
    #     # -P: Continue partials
    #     t = ParCommand(['rsync', '-aP', '--delete', '--perms', '--chmod=ugo+rwx', source, target], logger)
    #     t.start()
    #     return t
    # def task_remove(self, target):
    #     # Clean target first
    #     t = ParCommand(['rm', '-rf', target], logger)
    #     t.start()
    #     return t
    # def task_copy(self, source, target):
    #     t = ParCommand(['cp', '-r', source, target], logger)
    #     t.start()
    #     return t
    # def task_move(self, source, target):
    #     t = ParCommand(['mv', source, target], logger)
    #     t.start()
    #     return t

    # def take_memory_snapshop(self, memroot, name):
    #     target = os.path.join(self.state['path_memsnapshot'], name)
    #     t0 = self.task_remove(target)
    #     t0.join()
    #     t = self.task_move(source=memroot, target=target)
    #     # Join, because memory will keep changing during "rsync".
    #     t.join()
    # def load_memory_snapshot(self, memroot, name):
    #     source = os.path.join(self.state['path_memsnapshot'], name)
    #     t = self.task_copy(source=source+"/*", target=memroot+"/")
    #     # Join, because we cannot proceed without this task already completed.
    #     t.join()

    #################################
    # Apparatus for model save/load #
    #################################
    # TODO: Copying the visdom.log file to the current session for replaying.
    def load_states(self):
        filename = os.path.join(self.args["load_checkpoint"], "states.pt")
        logger.info("Loading states from file:" + filename)
        states = torch.load(filename, map_location=self.device)
        return states

    def load_runner(self):
        # If loading from a checkpoint, we must check the existence
        # of that path and whether that's a valid digideep session.
        # Existence is checked but validity is not. How is that?
        try:
            filename = os.path.join(self.args["load_checkpoint"], "runner.pt")
            logger.info("Loading runner from file:" + filename)
            runner = pickle.load(open(filename, "rb"))
        except Exception as ex:
            logger.fatal("Error loading from checkpoint:", ex)
            exit()
        return runner



    #################################

    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, value):
        self.state.update({key: value})

    def __delitem__(self, key):
        del self.state[key]

    def parse_arguments(self):
        # A bunch of arguments can come here!
        # These arguments are not saved!
        parser = argparse.ArgumentParser()
        ## Save/Load/Dry-run
        parser.add_argument('--load-checkpoint', metavar=('<path>'), default='', type=str,
                            help="Load a checkpoint to resume training from that point.")
        parser.add_argument('--play', action="store_true", help="Will play the stored policy.")
        parser.add_argument('--resume', action="store_true", help="Will resume training the stored policy.")
        parser.add_argument('--custom', action="store_true", help="Will load a checkpoint and run a custom runner.")
        parser.add_argument('--create-session-only', action="store_true",
                            help="If used, only the barebone session will be created and nt training/evaluation/loading will happen.")
        parser.add_argument('--dry-run', action="store_true",
                            help="If used, no footprints will be stored on disc whatsoever.")
        ## Session
        parser.add_argument('--session-path', metavar=('<path>'), default='log_sessions', type=str,
                            help="The path to store the sessions. Default is in /tmp")
        parser.add_argument('--session-name', metavar=('<name>'), default='', type=str,
                            help="A default name for the session. Random name if not provided.")
        parser.add_argument('--save-modules', metavar=('<path>'), default=[], nargs='+', type=str,
                            help="The modules to be stored in the session.")
        parser.add_argument('--log-level', metavar=('<n>'), default=1, type=int,
                            help="The logging level: 0 (debug and above), 1 (info and above), 2 (warn and above), 3 (error and above), 4 (fatal and above)")
        parser.add_argument('--profiler-level', metavar=('<n>'), default=-1, type=int,
                            help="Profiler level. '-1' profiles all level. Default: '-1'")
        parser.add_argument('--msg', metavar=('<msg>'), default='', type=str,
                            help="A message describing the current simulation and its significance.")
        ## Visdom Server
        parser.add_argument('--visdom', action='store_true', help="Whether to use visdom or not!")
        parser.add_argument('--visdom-port', metavar=('<n>'), default=8097, type=int,
                            help="The port of visdom server, it's on 8097 by default.")
        ## Tensorboard
        parser.add_argument('--tensorboard', action='store_true', help="Whether to use Tensorboard or not!")
        ## Monitor Thread
        parser.add_argument('--no-monitor-cpu', action="store_true",
                            help="Specify if you do not want CPU usage to be monitored.")
        parser.add_argument('--no-monitor-gpu', action="store_true",
                            help="Specify if you do not want GPU usage to be monitored.")
        # parser.add_argument('--monitor-interval', metavar=('<n>'), default=1, type=int, help="The interval for sampling cpu/gpu monitoring data.")
        # parser.add_argument('--monitor-window', metavar=('<n>'), default=10, type=int, help="The window size for sampling cpu/gpu monitoring data.")
        ## Parameters
        parser.add_argument('--params', metavar=('<name>'), default='', type=str, help="Choose the parameter set.")
        parser.add_argument('--cpanel', metavar=('<json dictionary>'), default=r'{}', type=JsonDecoder,
                            help="Set the parameters of the cpanel by a json dictionary.")

        parser.add_argument('--repeal', metavar=('<json dictionary>'), default=r'{}', type=JsonDecoder,
                            help="Set parameter values to be overridden.")
        # NOTE: No default value for params. MUST be specified explicitly. "digideep.params.mujoco"
        ##
        # parser.add_argument('--override', action='store_true', help="Provide this option to explicitly override saved options with new options.")
        # Override option should explicitly be set if you want to use "input-params".
        args = parser.parse_args()

        self.args = vars(args)