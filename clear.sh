#!/bin/sh

DATE=`date +%Y%m%d-%H%M%S`
FILE_NAME="log_$DATE.tar.gz"
tar -cvzf "$FILE_NAME" parameters.json log.txt log/* model/*
mv $FILE_NAME ~/ 
