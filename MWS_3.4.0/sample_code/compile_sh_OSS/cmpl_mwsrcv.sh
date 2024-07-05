#!/bin/bash

#compile command
echo "Compiling mwsrcv ..."
c11 \
../../src/*.cpp ../lib/src/*.cpp ../mwsrcv/src/*.cpp \
-o ../bin/mwsrcv \
-g \
-O2 \
-D _PUT_MODEL_=1 \
-D _TANDEM_SOURCE=1 \
-D _XOPEN_SOURCE=1 \
-D _XOPEN_SOURCE_EXTENDED=1 \
-D HAVE_CONFIG_H=1 \
-D HAVE_PTHREAD_H=1 \
-D SOCKLEN_T=1 \
-l WPUTDLL \
-W extensions \
-W lp64 \
-W saveabend \
-W systype=oss

rm *.o
