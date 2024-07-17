#!/bin/bash
rel_script_dir=$(dirname ${BASH_SOURCE[0]})
script_dir=$(readlink -f -- $rel_script_dir)
cd $script_dir

eval `ssh-agent` && ssh-add

#import other scripts in the framework
. ./dir_utils.sh
. ./runner.sh
. ./network.sh

benchmark_dirs=($(get_benchmark_directories ..))

for dir in ${benchmark_dirs[@]}
do
    run_benchmark $dir
done

wait_for_done ${benchmark_dirs[@]}

mkdir ../results
for dir in ${benchmark_dirs[@]}
do
    mv_results $dir "../results"
    mv_logs $dir "../results"
done

results=$(get_result_files ..)
email_results $results

archive_results "../"

clean "../"

ssh-agent -k
