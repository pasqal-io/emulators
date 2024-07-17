#activate benchmark venv and update all packages
make_sbatch_script(){
    benchmark_name=$(basename $1)
    folder=/scratch/$USER/$benchmark_name

    echo "#!/bin/bash" > sbatch.sh
    cat sbatch.config >> sbatch.sh
    echo "" >> sbatch.sh
    echo 'eval `ssh-agent` && ssh-add' >> sbatch.sh
    echo "mkdir $folder" >> sbatch.sh
    echo "cp -r /home/$USER/.julia $folder" >> sbatch.sh
    echo "python3 -m venv $folder/venv" >> sbatch.sh
    echo "cp /home/emuteam/.venv/pip.conf $folder/venv/pip.conf" >> sbatch.sh
    echo ". $folder/venv/bin/activate" >> sbatch.sh
    echo "pip install -e ../framework" >> sbatch.sh
    echo "pip install -e ../.." >> sbatch.sh
    echo "python3 benchmark.py" >> sbatch.sh
    echo "rm -rf $folder" >> sbatch.sh
    echo "ssh-agent -k" >> sbatch.sh
}

#calls benchmark.py in directory $1
run_benchmark(){
    cd $1
    make_sbatch_script $1
    sbatch sbatch.sh
    rm sbatch.sh
}
