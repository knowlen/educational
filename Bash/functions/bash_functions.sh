# Author: Nick Knowles (knowlen@wwu.edu)
#
# call this script with 2 command line args that aren't 
# "var1" and "var2" for a more complete demo.
# eg; $./bash_functions.sh 1.09 cat



#this is a bash function
some_name (){
    first_variable=$1 # this is how you access the first 
                      # variable passed to a bash function. 
    
    second_variable=$2 # and the second variable

    # do stuff
    echo "in some_name function"
    for i in {1..10}; do echo "$i: $1 $2"; done 
}


some_name "var1" "var2" # this is how you call a bash function 
                        # with 2 variables.

some_name $1 $2 # this is how you'd pass the 1st and 2nd 
                # command line args to a function. $1, $2, $3 ect.. 
                # in global scope = command line args. 

some_name # this is how you'd call a bash function with 
          # no variables



