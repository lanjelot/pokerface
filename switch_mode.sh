#!/bin/bash
MODEFILE=$HOME/code/pokerface.git/mode.txt
case $(cat $MODEFILE) in
  auto) echo manual > $MODEFILE
    ;;
  *) echo auto > $MODEFILE
    ;;
esac
    
