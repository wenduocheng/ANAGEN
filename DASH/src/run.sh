#!/bin/bash

python3 -W ignore main.py --dataset deepsea --arch deepsea --experiment_id 0 --valid_split 0

# speed test
# python3 speed.py --experiment_id 0
# python3 speed.py --experiment_id 0 --test_input_size 1
