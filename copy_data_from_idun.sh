#!/bin/sh
# Copy data from experiment on idun to local machine

scp -r trymte@idun.hpc.ntnu.no:$IDUN_DATADIR/$1 ~/Desktop/machine_learning/rlmpc/$1