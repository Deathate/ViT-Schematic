#!/bin/bash

#compile command
echo "Compiling mwsrcv ..."
g++ \
../../src/*.cpp ../lib/src/*.cpp ../mwsrcv/src/*.cpp \
-o ../bin/mwsrcv \
-std=c++0x \
-m64 \
-g \
-O3 \
-l pthread \
-Wall

rm *.o
