# Author: Nick Knowles (knowlen@wwu.edu)

# rec
#  Takes an input, then does a recursive subtraction of 
#  1 until that input is 0.
#
rec (){
    if [ "$1" == "0" ]; then
        return; #this is how you stop a function in bash
    else
        echo $1
        rec $(($1 - 1)) #this is how you do math in bash 
    fi

}

# trying
#  prompts the user for input, then either exits or 
#  calls itself again depending on that input. 
#
trying (){
    echo "go? [y]"
    read x #this is how you read input from user in bash
    if [ "$x" == "y" ]; then
        echo "going"
        trying #this is how you call a function in bash
    else
        return
    fi
}

rec 7 # this calls the function rec passing input 7
trying

