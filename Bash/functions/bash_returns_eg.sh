# Author: Nick Knowles (knowlen@wwu.edu)


# NOTEs: 
#   $(( .. )) allows you to preform math ops.
#   $(..) spawns a sub-shell to execute commands/functions.   
#
#   Further readings:
#   http://www.linuxjournal.com/content/return-values-bash-functions



# broken_add
#  Return statements in Bash functions are reserved for exit 
#  status values (0 or 1).
#  Thus, this function does not work. 
broken_add (){
    return $(( $1 + $2 )) #this isn't a thing in bash. 
}



# bad_add
#  Store it in a global, this works but is bad practice & has cons:
#   -Takes extra work if multiple variables need to store different 
#    return values of the function. 
#   -Hard to debug.
global_return=
bad_add(){
    temp=$(( $1 + $2 ))
    global_return=$temp #$(( $1 + $2 ))
}



# good_add
#  Make it print to stdout, then store that output as
#  a variable in whatever scope you're calling the function.
#  The standard way of doing function returns in Bash. 
# eg; stored_return=$(good_add 2 3)
good_add (){
    echo $(( $1 + $2 )) 
}

asd=$(good_add 2 3)
echo "Good: $asd"

bad_add 2 3
echo "Bad: $global_return"

asdf=$(broken_add 2 3)
echo "broke: $asdf"


