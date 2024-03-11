# Project and experiment
project_dir=~/model-permutation
scratch_dir="/work3/jovje/model-permutation"
experiment="resnet"
sub_experiment="02_fashion_mnist"

# Model type and directories
model="resnet"
data="fashion_mnist"
model_dir="experiments/${experiment}/${sub_experiment}/models_${model}"
log_dir="logs/${experiment}/${sub_experiment}/models_${model}"

# Create directory trees
mkdir -p "${project_dir}/${model_dir}"
mkdir -p "${scratch_dir}/${model_dir}"
mkdir -p "${project_dir}/${log_dir}"
mkdir -p "${scratch_dir}/${log_dir}"

# Create symbolic links to scratch directory
if [ ! -L "${project_dir}/${model_dir}" ]; then
    rmdir "${project_dir}/${model_dir}"
    ln -s "${scratch_dir}/${model_dir}" "${project_dir}/${model_dir}" 
fi
if [ ! -L "${project_dir}/${log_dir}" ]; then
    rmdir "${project_dir}/${log_dir}"
    ln -s "${scratch_dir}/${log_dir}" "${project_dir}/${log_dir}" 
fi

# Point to scratch directory for saving things
model_dir="${scratch_dir}/${model_dir}"
log_dir="${scratch_dir}/${log_dir}"

# BSUB arguments
queue="gpua100"
n_cores="15"
wall_time="12:00"
memory="1GB"
base_bsub_args=`echo \
    -q ${queue} \
    -gpu "num=1:mode=exclusive_process" \
    -n ${n_cores} \
    -R "span[hosts=1]" \
    -W ${wall_time} \
    -R "rusage[mem=${memory}]"`

base_command="\
    cd ~/;\
    source .virtualenvs/perm_gmm/bin/activate;\
    cd ${project_dir};\
    python3 ${project_dir}/experiments/${experiment}/train.py"

seeds=(1 2 3 4 5)

# Build commands
command_list=()
bsub_args_list=()
for seed in "${seeds[@]}"; do
    model_name="${model}_seed_${seed}"
    full_command="${base_command} \
        seed=${seed} \
        data=${data} \
        model=${model} \
        model_dir=${model_dir} \
        model_name=${model_name}.pt \
        model.hparams.pool_kernel_size=7"

    command_list+=("${full_command}")

    job_name=`echo ${experiment}_${sub_experiment}_${model_name}`
    std_out=`echo ${log_dir}/${model_name}.out`
    std_err=`echo ${log_dir}/${model_name}.err`   
    bsub_args=`echo ${base_bsub_args} -J ${job_name} -o ${std_out} -e ${std_err}`
    bsub_args_list+=("${bsub_args}")
done

# Execute commands
for ((i = 0; i < ${#command_list[@]}; i++)); do
    full_command="${command_list[$i]}"
    bsub_args="${bsub_args_list[$i]}"
    
    if [[ $1 = "bsub" ]]
    then
        echo "Now submitting to BSUB:"
        echo ${full_command}
        bsub $bsub_args $full_command
        sleep 1
    # Run interactively
    else
        echo "Now running:"
        echo ${full_command}
        eval ${full_command}
    fi
done