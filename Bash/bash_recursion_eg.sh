#!/usr/bin/env bash


# rec
#  Takes an input, then does a recursive subtraction of 
#  1 until that input is 0.
#
rec (){
    if [ "$1" == "0" ]; then
        exit;
    else
        echo $1
        rec $(($1 - 1)) 
    fi

}

# trying
#  prompts the user for input, then either exits or 
#  calls itself again depending on that input. 
#
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

