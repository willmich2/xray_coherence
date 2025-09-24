import time
import os
import copy
import numpy as np # type: ignore
from datetime import datetime
from src.threshold_opt import x_I_opt


def threshold_opt_sweep(design_dicts):
    opt_xs = []
    opt_x_fulls = []
    opt_Is = []
    opt_objs = []
    obj_values = []
    x_values = []
    start_time = time.time()
    for i, design_dict in enumerate(design_dicts):
        opt_x, opt_x_full, opt_I, opt_obj, obj_vals, x_vals = x_I_opt(design_dict)
        
        opt_xs.append(opt_x)
        opt_x_fulls.append(opt_x_full)
        opt_Is.append(opt_I)
        opt_objs.append(opt_obj)
        obj_values.append(obj_vals)
        x_values.append(x_vals)

        curr_time = time.time()
        print(f"finished design dict {i}")
        print(f"time elapsed: {round(curr_time - start_time)} s")
        print("----------------------------------")
        
    return opt_xs, opt_x_fulls, opt_Is, opt_objs, obj_values, x_values


def threshold_opt_iterative(design_dicts):
    opt_xs = []
    opt_x_fulls = []
    opt_Is = []
    opt_objs = []
    obj_values = []
    x_values = []

    start_time = time.time()
    for i, design_dict in enumerate(design_dicts):
        
        if i == 0:
            opt_x, opt_x_full, opt_I, opt_obj, obj_vals, x_vals = x_I_opt(design_dict)
        else: 
            design_dict_diffx = copy.deepcopy(design_dict) 
            design_dict_diffx["x_init"] = opt_xs[-1]
            opt_x, opt_x_full, opt_I, opt_obj, obj_vals, x_vals = x_I_opt(design_dict)
        
        opt_xs.append(opt_x)
        opt_x_fulls.append(opt_x_full)
        opt_Is.append(opt_I)
        opt_objs.append(opt_obj)
        obj_values.append(obj_vals)
        x_values.append(x_vals)

        del opt_x
        del opt_x_full
        del opt_I
        del opt_obj
        del obj_vals
        del x_vals

        curr_time = time.time()
        print(f"finished design dict {i}")
        print(f"time elapsed: {round(curr_time - start_time)} s")
        print("----------------------------------")
        
    return opt_xs, opt_x_fulls, opt_Is, opt_objs, obj_values, x_values


def get_formatted_datetime():
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H.%M.%S")
    # remove dashes and periods
    formatted_datetime = formatted_datetime.replace("-", "").replace(".", "")
    # replace colons with underscores
    formatted_datetime = formatted_datetime.replace(":", "_")
    # remove spaces
    formatted_datetime = formatted_datetime.replace(" ", "_")
    return formatted_datetime


def threshold_opt_sweep_save(design_dicts, sweep_func, path_string=""):
    opt_xs, opt_x_fulls, opt_Is, opt_objs, obj_values, x_values = sweep_func(design_dicts)
    end_time = get_formatted_datetime()

    if path_string == "":
        path_id = end_time
    else:
        path_id = path_string + end_time
    
    base_dir = f'/home/gridsan/wmichaels/opt_out/{path_id}'
    os.makedirs(base_dir, exist_ok=True)

    for arr, string in zip([opt_xs, opt_x_fulls, opt_Is, opt_objs, obj_values, x_values], ["xs", "x_fulls", "Is", "objs", "obj_values", "x_values"]):
        file_path = f'{base_dir}/{string}'
        np.savez(file_path, *arr)
    
    dict_file_path = f'{base_dir}/dict'
    np.save(dict_file_path, design_dicts)