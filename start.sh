#!/bin/bash

echo 'kill previous process'
ps -ef | grep one_model | grep -v grep | awk '{print $2}' | xargs kill -9
rm -rf nohup.out
echo 'start one_model process'

cd /opt/product/one_model

if [ $# -eq 1 ] && [ "$1" = "fg" ]; then
    echo "foreground start"
    python one_model/app.py
else
    echo "one_model background start"
    nohup python one_model/app.py &
fi
