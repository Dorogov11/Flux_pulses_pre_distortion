import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import dlsim, convolve

from pulses import (
    square_pulse,
    square_generic,
    constant_cosine,
    constant_cosine_reset,
    gaussian_pulse,
    double_square,
    double_square_ramped
)
from utils import PulseType, SweepType

filter_name = "Large_1"
#EXISTING_FILTER_PATH = "filters/" + filter_name + ".pickle"
EXISTING_FILTER_PATH = "E:/CQT/Fast Flux/Pre-distortion/filters/" + filter_name + ".pickle"




### Saving options ###
SAVE_PREDISTORTED_PULSE = 0
SAVE_PATH = "numerical_pulses/"
SAVE_PATH_direct_pulse = "ready_to_go_pulses/"

### Pulse generation options ###
PULSE_TYPE = PulseType.SQUARE_HOLD
# PULSE_TYPE = PulseType.CUSTOM
BATCH_NORMALIZATION = False

## Construct parameter sweep ##
# NOTE: ALL the params needed to construct the pulse must be defined in either CONSTANT_PARAMS or SWEEP_PARAM. See the individual generate_<pulse_type>_pulse functions for inofrmation on what parameters are needed.
# DO_PARAM_SWEEP = {"True"}
# CONSTANT_PARAMS = {"on_time": 20, "hold_time": 1200,}  # 1800
# SWEEP_PARAM = {
#     "name": "lpad",
#     "type": SweepType.VALUES,
#     "values": np.arange(40, 540, 100),  # [2000, 4000, 5000], #ns
# }
DO_PARAM_SWEEP = False
CONSTANT_PARAMS = {"on_time": 7_000, "hold_time": 1_000,}  # 1800
SWEEP_PARAM = {}

## Use your own pulse parameters ##
# NOTE: This will be ignored if DO_PARAM_SWEEP is set to True
PULSE_PARAMS = [
    {
        "lpad": 10,
        "on_time": 8_000,  # ns
        "hold_time": 1_000,
    },
]

# PULSE_PARAMS = [
#     {
#         "lpad": 5000,
#         "t1": 2000,
#         "t2": 10_000,
#         "outer_ramp":400,
#         "inner_ramp":200,
#         "rpad": 5000,
#         "A": 0.0,
#         "B": 10,
#     },
# ]
# PULSE_PARAMS = [
#     {
#         "path": "numerical_pulses/Fixed_L2_15_us.npz",
#     },
# ]

### Misc Options ###
SAMPLING_PERIOD = 1e-9  # It is unlikely that this should ever be changed
PLOT_PREDISTORTED_PULSE = 1


def predistort_pulse(iir_filters, fir_filters, pulse, norm=False):
    pulse_time_points = np.arange(len(pulse)) * SAMPLING_PERIOD

    for i in range(len(iir_filters)):
        curr_filter = iir_filters[i]["correction"]
        _, pulse = dlsim(curr_filter, pulse, t=pulse_time_points)

    # No FIR here
    # for i in range(len(fir_filters)):
    #     pulse = np.squeeze(pulse)
    #     filter_values = fir_filters[i]
    #     pulse = convolve(pulse, filter_values, mode="same")

    if norm:
        pulse = pulse / (
            max(abs(pulse))
        )  # normalize pulse as OPX only allows values from 0-1

    return pulse


def batch_normalize_pulses(pulse_list):
    """
    Renormalizing each pulse individually changes the target amp of the pulses. In order to make sure that they're the same across pulses generated in the same sweep, we normalize based on the max amplitude of all the pulses generated during the sweep.
    NOTE: We can't just use np.max since the pulses will have different lengths so we need to get the max of each pulse and then extract the max from there.
    """
    max_amp_list = [max(pulse) for pulse in pulse_list]
    max_amp = max(max_amp_list)
    normalized_pulses = [pulse / max_amp for pulse in pulse_list]
    return normalized_pulses


def generate_square_pulse(pulse_params):
    on_time = pulse_params["on_time"]
    lpad = pulse_params["lpad"]
    hold_time = pulse_params["hold_time"]
    pulse = square_pulse(on_time=on_time, lpad=lpad, rpad=hold_time)
    return pulse


def generate_square_reset_pulse(pulse_params):
    on_time = pulse_params["on_time"]
    lpad = pulse_params["lpad"]
    rpad = pulse_params["on_time"]
    amp = [pulse_params["start_amp"], pulse_params["hold_amp"], pulse_params["end_amp"]]
    pulse = square_generic(on_time=on_time, lpad=lpad, rpad=rpad, amp=amp)
    return pulse


def generate_constant_cosine_pulse(pulse_params):
    length_constant = pulse_params["length_constant"]
    length_ring = pulse_params["length_ring"]
    lpad = pulse_params["lpad"]
    rpad = pulse_params["rpad"]
    pulse = constant_cosine(
        length_constant=length_constant, length_ring=length_ring, lpad=lpad, rpad=rpad
    )
    return pulse


def generate_constant_cosine_reset(pulse_params):
    length_constant = pulse_params["length_constant"]
    length_ring = pulse_params["length_ring"]
    lpad = pulse_params["lpad"]
    rpad = pulse_params["rpad"]
    pulse = constant_cosine_reset(
        length_constant=length_constant, length_ring=length_ring, lpad=lpad, rpad=rpad
    )
    return pulse


def generate_gaussian_pulse(pulse_params):
    sigma = pulse_params["sigma"]
    chop = pulse_params["chop"]
    lpad = pulse_params["lpad"]
    rpad = pulse_params["rpad"]
    pulse = gaussian_pulse(sigma=sigma, chop=chop, lpad=lpad, rpad=rpad)
    return pulse

def generate_double_square_pulse(pulse_params):
    t1 = pulse_params["t1"]
    t2 = pulse_params["t2"]
    lpad = pulse_params["lpad"]
    rpad = pulse_params["rpad"]
    A = pulse_params["A"]
    B = pulse_params["B"]
    pulse = double_square(t1=t1, t2=t2, lpad=lpad, rpad=rpad, A=A, B=B)
    return pulse

def generate_double_square_ramped_pulse(pulse_params):
    t1 = pulse_params["t1"]
    t2 = pulse_params["t2"]
    inner_ramp=pulse_params["inner_ramp"]
    outer_ramp=pulse_params["outer_ramp"]
    lpad = pulse_params["lpad"]
    rpad = pulse_params["rpad"]
    A = pulse_params["A"]
    B = pulse_params["B"]
    pulse = double_square_ramped(t1=t1, t2=t2, inner_ramp=inner_ramp, outer_ramp=outer_ramp,lpad=lpad, rpad=rpad, A=A, B=B)
    return pulse


def load_existing_pulse(pulse_params):
    path = pulse_params["path"]
    pulse_file = np.load(path)
    pulse = pulse_file["I_quad"].flatten()
    return pulse


def generate_predistorted_pulse(
    pulse_type, iir_filters, fir_filters, pulse_params, norm=False
):
    if pulse_type == PulseType.SQUARE_HOLD:
        _, pulse = generate_square_pulse(pulse_params=pulse_params)
    elif pulse_type == PulseType.SQUARE_RESET:
        _, pulse = generate_square_reset_pulse(pulse_params=pulse_params)
    elif pulse_type == PulseType.GAUSSIAN:
        _, pulse = generate_gaussian_pulse(pulse_params=pulse_params)
    elif pulse_type == PulseType.CONSTANT_COSINE:
        _, pulse = generate_constant_cosine_pulse(pulse_params=pulse_params)
    elif pulse_type == PulseType.CONSTANT_COSINE_RESET:
        _, pulse = generate_constant_cosine_reset(pulse_params=pulse_params)
    elif pulse_type == PulseType.CUSTOM:
        pulse = load_existing_pulse(pulse_params=pulse_params)
    elif pulse_type == PulseType.DOUBLE_SQUARE:
        _, pulse = generate_double_square_pulse(pulse_params=pulse_params)
    elif pulse_type == PulseType.DOUBLE_SQUARE_RAMPED:
        _, pulse = generate_double_square_ramped_pulse(pulse_params=pulse_params)
    else:
        raise (f"Pulse type {pulse_type} not yet implemented")

    predistorted_pulse = predistort_pulse(iir_filters, fir_filters, pulse, norm)
    return predistorted_pulse


def build_pulse_param_list(constant_params, sweep_params):
    sweep_type = sweep_params["type"]
    sweep_values = sweep_params["values"]
    sweep_param_name = sweep_params["name"]
    if sweep_type == SweepType.RANGE:
        start = sweep_values[0]
        step = sweep_values[2]
        end = sweep_values[1] + step
        sweep_values = list(range(start, end, step))

    param_list = [None] * len(sweep_values)
    for i in range(len(sweep_values)):
        curr_param = {sweep_param_name: sweep_values[i]}
        for constant_name in constant_params.keys():
            curr_param[constant_name] = constant_params[constant_name]
        param_list[i] = curr_param
    return param_list


if __name__ == "__main__":
    print("--------- LOADING EXISTING FILTERS ----------")
    with open(EXISTING_FILTER_PATH, "rb") as filter_file:
        existing_filters = pickle.load(filter_file)
    iir_filters = existing_filters["iir_filters"]
    fir_filters = existing_filters["fir_filters"]

    print("--------- GENERATING PREDISTORTED PULSES ----------")
    if DO_PARAM_SWEEP:
        pulse_param_list = build_pulse_param_list(CONSTANT_PARAMS, SWEEP_PARAM)
    else:
        pulse_param_list = PULSE_PARAMS

    num_pulses = len(pulse_param_list)
    pulse_list = [None] * num_pulses
    do_indv_normalization = not (BATCH_NORMALIZATION)
    for i in range(num_pulses):
        print(pulse_param_list[i])
        pulse_list[i] = generate_predistorted_pulse(
            pulse_type=PULSE_TYPE,
            iir_filters=iir_filters,
            fir_filters=fir_filters,
            pulse_params=pulse_param_list[i],
            norm=do_indv_normalization,
        )

    if BATCH_NORMALIZATION:
        pulse_list = batch_normalize_pulses(pulse_list=pulse_list)

    for i in range(num_pulses):
        predistorted_pulse = pulse_list[i]
        if PLOT_PREDISTORTED_PULSE:
            pulse_time_points = np.arange(len(predistorted_pulse)) * SAMPLING_PERIOD
            print(predistorted_pulse)
            plt.plot(pulse_time_points, predistorted_pulse, color="blue")
            plt.show()

        if SAVE_PREDISTORTED_PULSE:
            save_location = f"{SAVE_PATH}/{i}.npz"
            if DO_PARAM_SWEEP:
                save_location = f"{SAVE_PATH}/square_IIR_FIR_{SWEEP_PARAM['name']}_{pulse_param_list[i][SWEEP_PARAM['name']]}ns.npz"
            np.savez(
                save_location,
                I_quad=predistorted_pulse,
                Q_quad=predistorted_pulse,
                dt=[1],
            )
            print(f"Waveform saved to {save_location}")

            save_location_ = f"{SAVE_PATH_direct_pulse}/{filter_name}_15us_double_filter.npy"
            if DO_PARAM_SWEEP:
                save_location_ = f"{SAVE_PATH}/square_IIR_FIR_{SWEEP_PARAM['name']}_{pulse_param_list[i][SWEEP_PARAM['name']]}ns.npy"
            np.save(
                save_location_,
                predistorted_pulse.flatten(),
            )
            print(f"Sexy Waveform saved to {save_location_}")
    print("---------------------- DONE -----------------------")
