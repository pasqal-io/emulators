# add sbatch and julia path for crontab
#export PATH=/softs/batch/slurm/current/bin:${PATH}
#export PATH=/softs/devel/julia/1.8.5/bin:${PATH}

home_dir="/home/$USER"

cd $home_dir

# Specify the directory to check for
dir="$home_dir/benchmarks"

# Check if the directory exists and clean if so
if ! [ -d $dir ]
then
    mkdir $dir
else
    rm -r --interactive=never -- "$dir"
    mkdir $dir
fi

cd $dir

eval $(ssh-agent) && ssh-add
/usr/bin/git clone git@gitlab.pasqal.com:emulation/rydberg-atoms/emu-ct.git
cd emu-t
/usr/bin/git checkout BRANCH

#TODO replace with sbatch line when mail works on compute nodes
nohup ./benchmarks/framework/benchmarks.sh > /dev/null 2> /dev/null < /dev/null &
#sbatch -c1 "benchmarks/framework/benchmarks.sh"
