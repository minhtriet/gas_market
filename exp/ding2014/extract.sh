#!/usr/bin/env bash
echo "Working with $1"
java -Xmx512m -jar exp/ding2014/reverb-latest.jar $1 > $2
