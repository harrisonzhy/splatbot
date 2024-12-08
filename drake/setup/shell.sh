#!/bin/bash

sinteractive -p wholenode \
             --account=<account-id>-gpu \
             --nodes=1 --ntasks-per-node=1 \
             --gpus-per-node=1 \
             --mem-per-gpu=40G \
             --time=04:00:00 \
             -p gpu

