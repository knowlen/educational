#!/usr/bin/env bash

rec (){
    if [ "$1" == "0" ]; then
        exit;
    else
        echo $1
        rec $(($1 - 1)) 
    fi

}

trying (){
    echo "go?"
    read x
    if [ "$x" == "y" ]; then
        echo "going"
        trying
    else
        exit
    fi

}


trying

