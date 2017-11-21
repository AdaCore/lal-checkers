#! /bin/sh

export PYTHONPATH=$PWD:$PYTHONPATH
exec ./testsuite.py $@
