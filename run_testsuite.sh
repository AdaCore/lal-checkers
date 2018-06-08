#! /bin/sh

export PYTHONPATH=$PWD:$PYTHONPATH
exec ./testsuite/testsuite.py $@
