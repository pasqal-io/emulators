#returns "true" if all files in array $1 exist, "false" otherwise
all_exist(){
    for file in $@
    do
        if [[ ! -f $file  ]]
	then
	    echo false
	    return 0
	fi
    done
    echo true
}

#returns all subdirs of $1, excluding "framework", "benchmark-venv" and "results"
get_benchmark_directories(){
    echo $(find $1 -maxdepth 1 -mindepth 1 -type d -not -name "framework" -not -name "benchmark-venv" -not -name "results")
}

#move results from benchark $1 to $2/$1. results are assumed to be everything except the file named DONE
mv_results(){
    result_files=$(find "$1/results" -maxdepth 1 -type f -not -name "DONE")
    benchmark_name=$(basename $1)
    mkdir $2/$benchmark_name
    if [[ ! ${#result_files[@]} -eq 0 ]]
    then
        mv -t $2/$benchmark_name $result_files
    fi
}

#move .out files from benchmark $1 to $2. resulting out file is called $1.out
mv_logs(){
    cd $1
    benchmark_name=$(basename $1)
    rename slurm-*.out $1.out *.out
    mv -t $2/$benchmark_name $1.out
}

#archives the results folder in $1 to /home/$USER/archive/<timestamp>
archive_results(){
    timestamp=$(date +"%Y%m%dT%H%M")
    archivedir=/home/$USER/archive
    mkdir -p $archivedir
    mv -f $1/results $archivedir/$timestamp
}

#cleans up $1
clean(){
    for dir in ${benchmark_dirs[@]}
    do
	rm -r "$dir/results"
    done
}

#waits until /results/DONE exists in each dir in $1
wait_for_done(){
    done_files=( "${@/%//results/DONE}" )
    while [[ $(all_exist ${done_files[@]}) = false ]]
    do
        sleep 5
    done
}

#get all files in $1/results, excluding .out files assumed to be logs
get_result_files(){
    echo $(find $1/results -type f -not -name *.out)
}
