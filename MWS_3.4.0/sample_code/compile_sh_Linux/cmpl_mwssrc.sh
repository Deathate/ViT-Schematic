#!/bin/bash

#compile command
echo "Compiling mwssrc ..."
g++ \
../../src/*.cpp ../lib/src/*.cpp ../mwssrc/src/*.cpp \
-o ../bin/mwssrc \
-std=c++0x \
-m64 \
-g \
-O3 \
-l pthread \
-Wall

rm *.o
