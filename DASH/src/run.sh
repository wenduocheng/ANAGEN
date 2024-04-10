#!/bin/bash

python3 -W ignore main.py --dataset deepsea --arch wrn --experiment_id 1

# speed test
# python3 speed.py --experiment_id 0
# python3 speed.py --experiment_id 0 --test_input_size 1
