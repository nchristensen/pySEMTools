'''Functions for indexing of files'''
import re
import json
import os
import numpy as np
from ..monitoring.logger import Logger
import glob
from pymech.neksuite.field import read_header

def index_files_from_log(comm, logpath="", logname="", progress_reports=50):

    logger = Logger(comm=comm, module_name="file_index_from_log")

    if logpath == "":
        path = os.getcwd() + "/"
    else:
        path = logpath

    logger.write("info", f"Reading file: {path + logname}")
    logger.tic()
    logfile = open(path+logname, "r")
    log = logfile.readlines()
    number_of_lines = len(log)
    logfile.close()
    logger.toc()

    # Search patterns
    file_patterns = {"File name": re.compile(r"File name\s*:\s*(?P<value>.+)")}
    t_pattern = re.compile(r"t =\s*(?P<t_value>[+-]?\d*\.\d+E[+-]?\d+).*")
    output_number_pattern = re.compile(r"Output number\s*:\s*(?P<value>.+)")
    writing_time_pattern = re.compile(
        r"Writing at time:\s*(?P<writing_time>\d+\.\d+).*"
    )

    added_files = []

    logger.write("info", "Start reading the lines")
    report_interval = number_of_lines // progress_reports
    read_first_time_step = False
    for i, line in enumerate(log):




        if np.mod(i, report_interval) == 0:
            print(
                "========================================================================================="
            )
            logger.write("info", f"read up to line: {i}, out of {number_of_lines} <-> {i/number_of_lines*100}%")
            print(
                "========================================================================================="
            )

        if not read_first_time_step:

            for key, pattern in file_patterns.items():
                match = pattern.search(line)
                if match: 
                    add_file = match.group("value").strip()
                    if add_file not in added_files:
                        added_files.append(add_file)
        


        # Find the first time step
        if "Time-step" in line and not read_first_time_step:

            match = t_pattern.search(log[i - 2])
            if match:
                start_time = float(match.group("t_value"))

            logger.write("info", f"Run start time: {start_time}")

            read_first_time_step = True

            files = {}
            files_found = {}
            files_index = {}
            files_last_sample = {}
            for file in added_files:
                files[file] = dict()
                files[file]["simulation_start_time"] = start_time

            for file in added_files:
                files_found[file] = False

            for file in added_files:
                files_index[file] = 0

            for file in added_files:
                files_last_sample[file] = start_time

            lines_to_check = 2 * len(added_files) + 2

        if "Writer output" in line and read_first_time_step:

            print(
                "========================================================================================="
            )

            for file in added_files:
                files_found[file] = False

            writer_output_block = log[i : i + lines_to_check]

            # Check if this output block contains another output
            for j, next_line in enumerate(writer_output_block):
                if ("Writer output" in next_line and j > 0):
                    k = j
                    break
                else:
                    k = 10000

            for j, next_line in enumerate(writer_output_block):

                if j >= k:
                    break

                for file in added_files:
                    if " " + file in next_line:
                        files_found[file] = True
                        
                        match = output_number_pattern.search(writer_output_block[j + 1])
                        if match:
                            output_number = int(match.group("value"))
                            file_name = (
                                file.split(".")[0] + "0.f" + str(output_number).zfill(5)
                            )

                        files[file][files_index[file]] = dict()
                        files[file][files_index[file]]["fname"] = file_name

                        # Check if the file exists in this folder
                        file_exists = os.path.exists(path + file_name)
                        if file_exists:
                            files[file][files_index[file]]["path"] = os.path.abspath(
                                path + file_name
                            )
                        else:
                            files[file][files_index[file]][
                                "path"
                            ] = "file_not_in_folder"

                if "Writing at time" in next_line:

                    match = writing_time_pattern.search(next_line)
                    if match:
                        file_time = float(match.group("writing_time"))

            for file in added_files:
                if files_found[file]:
                    files[file][files_index[file]]["time"] = file_time
                    files[file][files_index[file]]["time_previous_output"] = files_last_sample[
                        file
                    ]
                    files[file][files_index[file]]["time_interval"] = (
                        file_time - files_last_sample[file]
                    )

                    files_last_sample[file] = file_time

                    logger.write("info", f"{file} found")
                    for key in files[file][files_index[file]].keys():
                        logger.write(
                            "info", f"{key}: {files[file][files_index[file]][key]}"
                        )
                    files_index[file] += 1

    print(
        "========================================================================================="
    )
    print(
        "========================================================================================="
    )

    logger.write("info", "Check finished")

    for file in added_files:
        logger.write("info", f"Writing {file} file index")
        logger.tic()
        with open(path + file + "_index.json", "w") as outfile:
            outfile.write(json.dumps(files[file], indent=4))
        logger.toc()

    del logger


def index_files_from_folder(comm, folder_path = "", run_start_time = 0, stat_start_time = 0):

    if folder_path == "":
        folder_path = os.getcwd() + "/"

    if folder_path[-1] != "/":
        folder_path = folder_path + "/"

    logger = Logger(comm=comm, module_name="file_index_from_folder")

    logger.write("info", f"Reading folder: {folder_path}")

    # Get all files in the folder that are fields
    files_in_folder = glob.glob(folder_path + "*0.f*")

    added_files = []
    for i in range(0, len(files_in_folder)):
        files_in_folder[i] = os.path.basename(files_in_folder[i])
        
        ftype = files_in_folder[i].split(".")[0][:-1] + ".fld"
        if ftype not in added_files:
            added_files.append(ftype)

    for ftype in added_files:
        logger.write("info", f"Found files with {ftype} pattern")
        
    print(
        "========================================================================================="
    )
 
    files = {}
    files_index = {}
    files_last_sample = {}
    for ftype in added_files:
        files[ftype] = dict()
        files[ftype]["simulation_start_time"] = run_start_time
        files_index[ftype] = 0

    for i, file_in_folder in enumerate(files_in_folder):

        logger.write("info", f"Indexing file: {file_in_folder}")

        ftype = files_in_folder[i].split(".")[0][:-1] + ".fld"

        files[ftype][files_index[ftype]] = dict()
        files[ftype][files_index[ftype]]["fname"] = file_in_folder
        files[ftype][files_index[ftype]]["path"] = os.path.abspath(folder_path + file_in_folder)

        # Determine the time from the header
        header = read_header(files[ftype][files_index[ftype]]["path"])
        current_time = header.time

        files[ftype][files_index[ftype]]["time"] = current_time

        if files_index[ftype] == 0:
            if ( "stat" in ftype or "mean" in ftype): 
                files[ftype][files_index[ftype]]["time_previous_output"] = stat_start_time
            else: 
                files[ftype][files_index[ftype]]["time_previous_output"] = run_start_time 
        else:
            files[ftype][files_index[ftype]]["time_previous_output"] = files[ftype][files_index[ftype]-1]["time"]

        files[ftype][files_index[ftype]]["time_interval"] = current_time - files[ftype][files_index[ftype]]["time_previous_output"]
                    
        for key in files[ftype][files_index[ftype]].keys():
            logger.write(
                "info", f"{key}: {files[ftype][files_index[ftype]][key]}"
            )

        files_index[ftype] += 1
            
        print(
            "========================================================================================="
        )

    logger.write("info", "Check finished")

    for file in added_files:
        logger.write("info", f"Writing {file} file index")
        logger.tic()
        with open(folder_path + file + "_index.json", "w") as outfile:
            outfile.write(json.dumps(files[file], indent=4))
        logger.toc()

    del logger