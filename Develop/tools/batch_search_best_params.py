import os
import sys
import glob
import subprocess

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
sys.path.append(ROOT)

from utils.logging_utils import config_logging


DATA_ROOT =  os.path.realpath(os.path.join(ROOT, "data"))
CLASSIFER = ["RF", "SVR", "BPNN"]
LOGGER = config_logging()
RUNNER_PATH = os.path.join(ROOT, "runer.py")

data_files = glob.glob(f"{DATA_ROOT}/*.csv", recursive=False)

for data_f in data_files:
    subproces_list = list()
    for classifer in CLASSIFER:
        args = ["python", RUNNER_PATH, "-i", data_f, "-c", classifer, "--split-ratio", "0.8", "--search-best"]
        args.extend(["--run-time", "1"])
        args.extend(["--search-metric", "R2"])
        title = os.path.splitext(os.path.split(data_f)[1])[0]
        args.extend(["-o", f"./result_params/{classifer}/{title}"])

        LOGGER.info(" ".join(args))
        p = subprocess.Popen(args, shell=False)
        subproces_list.append(p)

    for p in subproces_list:
        p.wait()