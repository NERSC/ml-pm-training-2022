# SC21 Deep Learning at Scale Tutorial

This repository contains the example code material for the SC21 tutorial:
*Deep Learning at Scale*.

**Contents**
* [Links](#links)
* [Installation](#installation-and-setup)
* [Model, data, and code overview](#model-data-and-training-code-overview)
* [Single GPU training](#single-gpu-training)
* [Single GPU performance](#single-gpu-performance-profiling-and-optimization)
* [Distributed training](#distributed-gpu-training)
* [Multi GPU performance](#multi-gpu-performance-profiling-and-optimization)
* [Putting it all together](#putting-it-all-together)

## Links

** TODO: update the slides links **

Tutorial slides: https://drive.google.com/drive/folders/1Ei56_HDjLMPbdLq9QdQhoxN3J1YdzZw0?usp=sharing

NERSC JupyterHub: https://jupyter.nersc.gov

Data download: https://portal.nersc.gov/project/dasrepo/pharring/

## Installation and Setup

### Software environment

Access to NERSC's Perlmutter machine is provided for this tutorial via [jupyter.nersc.gov](https://jupyter.nersc.gov). 
Training account setup instructions will be given during the session. Once you have your provided account credentials, you can log in to Jupyter via the link (leave the OTP field blank when logging into Jupyter).
Once logged into the hub, start a session by clicking the button for Perlmutter Shared CPU Node (other options will not work with this tutorial material). This will open up a session on a Perlmutter login node, from which you can submit jobs to the GPU nodes and monitor their progress.

To begin, start a terminal from JupyterHub and clone this repository with:
```bash
git clone https://github.com/NERSC/ml-pm-training-2022.git
```
You can use the Jupyter file browser to view and edit source files and scripts. For all of the example commands provided below, make sure you are running them from within the top-level folder of the repository. In your terminal, change to the directory with
```bash
cd ml-pm-training-2022
```

For running slurm jobs on Perlmutter, we will use training accounts which are provided under the `ntrain4` project. The slurm script `submit_pm.sh` included in the repository is configured to work automatically as is, but if you submit your own custom jobs via `salloc` or `sbatch` you must include the following flags for slurm:
* `-A ntrain4_g` is required for training accounts
* `--reservation=sc21_tutorial_01` is required to access the set of GPU nodes we have reserved for the duration of the tutorial.

The code can be run using the `romerojosh/containers:sc21_tutorial` docker container. On Perlmutter, docker containers are run via [shifter](https://docs.nersc.gov/development/shifter/), and this container is already downloaded and automatically invoked by our job submission scripts. Our container is based on the [NVIDIA ngc 21.10 pytorch container](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-10.html#rel_21-10), with a few additional packages added. See the dockerfile in [`docker/Dockerfile`](docker/Dockerfile) for details.

## Model, data, and training code overview

The model in this repository is adapted from a cosmological application of deep learning ([Harrington et al. 2021](https://arxiv.org/abs/2106.12662)), which aims to augment computationally expensive simulations by using a [U-Net](https://arxiv.org/abs/1505.04597) model to reconstruct physical fields of interest (namely, hydrodynamic quantities associated with diffuse gas in the universe):

![n-body to hydro schematic](tutorial_images/nbody2hydro.png)

The U-Net model architecture used in these examples can be found in [`networks/UNet.py`](networks/UNet.py). U-Nets are a popular and capable architecture, as they can extract long-range features through sequential downsampling convolutions, while fine-grained details can be propagated to the upsampling path via skip connections. This particular U-Net is relatively lightweight, to better accommodate our 3D data samples.

The basic data loading pipeline is defined in [`utils/data_loader.py`](utils/data_loader.py), whose primary components are:
* The `RandomCropDataset` which accesses the simulation data stored on disk, and randomly crops sub-volumes of the physical fields to serve for training and validation. For this repository, we will be using a crop size of 64^3
* The `RandomRotator` transform, which applies random rotations and reflections to the samples as data augmentations
* The above components are assembled to feed a PyTorch `DataLoader` which takes the augmented samples and combines them into a batch for each training step.

It is common practice to decay the learning rate according to some schedule as the model trains, so that the optimizer can settle into sharper minima during gradient descent. Here we opt for the cosine learning rate decay schedule, which starts at an intial learning rate and decays continuously throughout training according to a cosine function. This is handled by the `lr_schedule` routine defined in [`utils/__init__.py`](utils/__init__.py), which also has logic to implement learning rate scaling and warm-up for use in the [Distributed GPU training](#Distributed-GPU-training) section

As we will see in the [Single GPU performance profiling and optimization](#Single-GPU-performance-profiling-and-optimization) section, the random rotations add considerable overhead to data loading during training, and we can achieve performance gains by doing these preprocessing steps on the GPU instead using NVIDIA's DALI library. Code for this is found in [`utils/data_loader_dali.py`](utils/data_loader_dali.py).

The script to train the model is [`train.py`](train.py), which uses the following arguments to load the desired training setup:
```
--yaml_config YAML_CONFIG   path to yaml file containing training configs
--config CONFIG             name of desired config in yaml file
```

Based on the selected configuration, the train script will then:
1.  Set up the data loaders and construct our U-Net model, the Adam optimizer, and our L1 loss function.
2.  Loop over training epochs to run the training. See if you can identify the following key components: 
    * Looping over data batches from our data loader.
    * Applying the forward pass of the model and computing the loss function.
    * Calling `backward()` on the loss value to backpropagate gradients. Note the use of the `grad_scaler` will be explained below when enabling mixed precision.
    * Applying the model to the validation dataset and logging training and validation metrics to visualize in TensorBoard (see if you can find where we construct the TensorBoard `SummaryWriter` and where our specific metrics are logged via the `add_scalar` call).

Besides the `train.py` script, we have a slightly more complex [`train_graph.py`](train_graph.py)
script, which implements the same functionality with added capability for using the CUDA Graphs APIs introduced in PyTorch 1.10. This topic will be covered in the [Single GPU performance profiling and optimization](#Single-GPU-performance-profiling-and-optimization) section.

More info on the model and data can be found in the [slides](https://drive.google.com/drive/u/1/folders/1Ei56_HDjLMPbdLq9QdQhoxN3J1YdzZw0). If you are experimenting with this repository after the tutorial date, you can download the data from here: https://portal.nersc.gov/project/dasrepo/pharring/.
Note that you will have to adjust the data path in `submit_pm.sh` to point yor personal copy after downloading.


## Single GPU training

First, let us look at the performance of the training script without optimizations on a single GPU.

On Perlmutter for the tutorial, we will be submitting jobs to the batch queue. To submit this job, use the following command:
```
sbatch -n 1 ./submit_pm.sh --config=short --num_epochs 3
```
`submit_pm.sh` is a batch submission script that defines resources to be requested by SLURM as well as the command to run.
Note that any arguments for `train.py`, such as the desired config (`--config`), can be added after `submit_pm.sh` when submitting, and they will be passed to `train.py` properly.
When using batch submission, you can see the job output by viewing the file `pm-crop64-<jobid>.out` in the submission
directory. You can find the job id of your job using the command `squeue --me` and looking at the first column of the output.

For interactive jobs, you can run the Python script directly using the following command (**NOTE: please don't run training on the Perlmutter login nodes**):
```
python train.py --config=short --num_epochs 3
```
For V100 systems, you will likely need to update the config to reduce the local batch size to 32 due to the reduced memory capacity. Otherwise, instructions are the same.

This will run 3 epochs of training on a single GPU using a default batch size of 64.
See [`config/UNet.yaml`](config/UNet.yaml) for specific configuration details.
Note we will use the default batch size for the optimization work in the next section
and will push beyond to larger batch sizes in the distributed training section.

In the baseline configuration, the model converges to a loss of about `4.75e-3` on
the validation dataset in 10 epochs. This takes around 2 hours to run, so to save time we have already included an example TensorBoard log for the `base` config in the `example_logs` directory for you.
We want to compare our training results against the `base` config baseline, and TensorBoard makes this easy as long as all training runs are stored in the same place. 
To copy the example TensorBoard log to the scratch directory where our training jobs will output their logs, do
```
mkdir -p $SCRATCH/ml-pm-training-2022/logs
cp -r ./example_logs/base $SCRATCH/ml-pm-training-2022/logs
```

To view results in TensorBoard, open the [`start_tensorboard.ipynb`](start_tensorboard.ipynb) notebook and follow the instructions in it to launch a TensorBoard session in your browser. Once you have TensorBoard open, you should see a dashboard with data for the loss values, learning rate, and average iterations per second. Looking at the validation loss for the `base` config, you should see the following training curve:
![baseline training](tutorial_images/baseline_tb.png)

As our training with the `short` config runs, it should also dump the training metrics to the TensorBoard directory, and TensorBoard will parse the data and display it for you. You can hit the refresh button in the upper-right corner of TensorBoard to update the plots with the latest data.

## Single GPU performance profiling and optimization

If you are interested in learning how to profile and optimize your PyTorch
code on Perlmutter, please refer to our full SC21 tutorial material at
https://github.com/NERSC/ml-pm-training-2022#single-gpu-performance-profiling-and-optimization

### Full training with optimizations
Now you can run the full model training on a single GPU with our optimizations. For convenience, we provide a configuration with the optimizations already enabled. Submit the full training with:

```
sbatch -n 1 -t 40 ./submit_pm.sh --config=bs64_opt
```

## Distributed GPU training

Now that we have model training code that is optimized for training on a single GPU,
we are ready to utilize multiple GPUs and multiple nodes to accelerate the workflow
with *distributed training*. We will use the recommended `DistributedDataParallel`
wrapper in PyTorch with the NCCL backend for optimized communication operations on
systems with NVIDIA GPUs. Refer to the PyTorch documentation for additional details 
on the distributed package: https://pytorch.org/docs/stable/distributed.html

### Code basics

To submit a multi-GPU job, use the `submit_pm.sh` with the `-n` option set to the desired number of GPUs. For example, to launch a training with multiple GPUs, you will use commands like:
```
sbatch -n NUM_GPU submit_pm.sh [OPTIONS]
```
This script automatically uses the slurm flags `--ntasks-per-node 4`, `--cpus-per-task 32`, `--gpus-per-task 1`, so slurm will allocate one process for each GPU we request, and give each process 1/4th of the CPU resources available on a Perlmutter GPU node. This way, multi-node trainings can easily be launched simply by setting `-n` greater than 4.

*Question: why do you think we run 1 task (cpu process) per GPU, instead of 1 task per node (each running 4 GPUs)?*

PyTorch `DistributedDataParallel`, or DDP for short, is flexible and can initialize process groups with a variety of methods. For this code, we will use the standard approach of initializing via environment variables, which can be easily read from the slurm environment. Take a look at the `export_DDP_vars.sh` helper script, which is used by our job script to expose for PyTorch DDP the global rank and node-local rank of each process, along with the total number of ranks and the address and port to use for network communication. In the [`train.py`](train.py) script, near the bottom in the main script execution, we set up the distributed backend using these environment variables via `torch.distributed.init_proces_group`.

When distributing a batch of samples in DDP training, we must make sure each rank gets a properly-sized subset of the full batch. See if you can find where we use the `DistributedSampler` from PyTorch to properly partition the data in [`utils/data_loader.py`](utils/data_loader.py). Note that in this particular example, we are already cropping samples randomly form a large simulation volume, so the partitioning does not ensure each rank gets unique data, but simply shortens the number of steps needed to complete an "epoch". For datasets with a fixed number of unique samples, `DistributedSampler` will also ensure each rank sees a unique minibatch.

In `train.py`, after our U-Net model is constructed,
we convert it to a distributed data parallel model by wrapping it as:
```
model = DistributedDataParallel(model, device_ids=[local_rank])
```

The DistributedDataParallel (DDP) model wrapper takes care of broadcasting
initial model weights to all workers and performing all-reduce on the gradients
in the training backward pass to properly synchronize and update the model
weights in the distributed setting.

*Question: why does DDP broadcast the initial model weights to all workers? What would happen if it didn't?*

### Large batch convergence

To speed up training, we try to use larger batch sizes, spread across more GPUs,
with larger learning rates. The base config uses a batchsize of 64 for single-GPU training, so we will set `base_batch_size=64` in our configs and then increase the `global_batch_size` parameter in increments of 64 for every additional GPU we add to the distributed training. Then, we can take the ratio of `global_batch_size` and `base_batch_size` to decide how much to scale up the learning rate as the global batch size grows. In this section, we will make use of the square-root scaling rule, which multiplies the base initial learning rate by `sqrt(global_batch_size/base_batch_size)`. Take a look at [`utils/__init__.py`](utils/__init__.py) to see how this is implemented.

*Question: how do you think the loss curves would change if we didn't increase the learning rate at all as we scale up?*

*Question: what do you think would happen if we simply increased our learning rate without increasing batch size?*

As a first attempt, let's try increasing the batchsize from 64 to 512, distributing our training across 8 GPUs (thus two GPU nodes on Perlmutter). To submit a job with this config, do
```
sbatch -t 10 -n 8 submit_pm.sh --config=bs512_test
```

Looking at the TensorBoard log, we can see that the rate of convergence is increased initially, but the validation loss plateaus quickly and our final accuracy ends up worse than the single-GPU training:
![batchsize 512 bad](tutorial_images/bs512_short.png)

From the plot, we see that with a global batch size of 512 we complete each epoch in a much shorter amount of time, so training concludes rapidly. This affects our learning rate schedule, which depends on the total number of steps as set in `train.py`:
```
params.lr_schedule['tot_steps'] = params.num_epochs*(params.Nsamples//params.global_batch_size)
```

If we increase the total number of epochs, we will run longer (thus giving the model more training iterations to update weights) and the learning rate will decay more slowly, giving us more time to converge quickly with a larger learning rate. To try this out, run the `bs512_opt` config, which runs for 40 epochs rather than the default 10:
```
sbatch -t 20 -n 8 submit_pm.sh --config=bs512_opt
```
With the longer training, we can see that our higher batch size results are slightly better than the baseline configuration. Furthermore, the minimum in the loss is reached sooner, despite running for more epochs:
![batchsize 512 good](tutorial_images/bs512.png)

Based on our findings, we can strategize to have trainings with larger batch sizes run for half as many total iterations as the baseline, as a rule of thumb. You can see this imlemented in the different configs for various global batchsizes: `bs256_opt`, `bs512_opt`, `bs2048_opt`. However, to really compare how our convergence is improving between these configurations, we must consider the actual time-to-solution. To do this in TensorBoard, select the "Relative" option on the left-hand side, which will change the x-axis in each plot to show walltime of the job (in hours), relative to the first data point:

![relative option for tensorboard](tutorial_images/relative.png)

With this selected, we can compare results between these different configs as a function of time, and see that all of them improve over the baseline. Furthermore, the rate of convergence improves as we add more GPUs and increase the global batch size:

![comparison across batchsizes](tutorial_images/bs_compare.png)

Based on our study, we see that scaling up our U-Net can definitely speed up training and reduce time-to-solution. Compared to our un-optimized single-GPU baseline from the first section, which took around 2 hours to train, we can now converge in about 10 minutes, which is a great speedup! We have also seen that there are several considerations to be aware of and several key hyperparameters to tune. We encourage you to now play with some of these settings and observe how they can affect the results. The main parameters in `config/UNet.yaml` to consider are:

* `num_epochs`, to adjust how long it takes for learning rate to decay and for training to conclude.
* `lr_schedule`, to choose how to scale up learning rate, or change the start and end learning rates.
* `global_batch_size`. We ask that you limit yourself to a maximum of 8 GPUs initially for this section, to ensure everyone gets sufficient access to compute resources.

You should also consider the following questions:
* *What are the limitations to scaling up batch size and learning rates?*
* *What would happen to the learning curves and runtime if we did "strong scaling" instead (hold global batch size fixed as we increase GPUs, and respectively decrease the local batch size)?*

## Multi-GPU performance profiling and optimization

With distributed training enabled and large batch convergence tested, we are ready 
to optimize the multi-GPU training throughput. We start with understanding and ploting
the performance of our application as we scale. Then we can go in more details and profile 
the multi-GPU training with Nsight Systems to understand the communication performance. 

### Weak and Strong Throughput Scaling

First we want to measure the scaling efficiency. An example command to generate the points for 8 nodes is:
```
BENCHY_OUTPUT=weak_scale sbatch -N 8 ./submit_pm.sh --num_data_workers 4 --local_batch_size 64 --config=bs64_opt --enable_benchy
```

<img src="tutorial_images/scale_perfEff.png" width="500">

The plot shows the throughput as we scale up to 32 nodes. The solid green line shows the real data throughput, while the dotted green line shows the ideal throughput, i.e. if we multiply the single GPU throughput by the number of GPUs used. For example for 32 nodes we get around 78% scaling efficiency. The blue lines show the data throughput by running the data-loader in isolation. The orange lines show the throughput for synthetic data.

Next we can further breakdown the performance of the applications, by switching off the communication between workers. An example command to generate the points for 8 nodes and adding the noddp flag is:
```
BENCHY_OUTPUT=weak_scale_noddp sbatch -N 8 ./submit_pm.sh --num_data_workers 4 --local_batch_size 64 --config=bs64_opt --enable_benchy --noddp
```

<img src="tutorial_images/scale_perfComm.png" width="500">

The orange line is with synthetic data, so no I/O overhead, and the orange dotted line is with synthetic data but having the communication between compute switched off. That effectively makes the dotted orange line the compute of the application. By comparing it with the solid orange line we can get the communication overhead. For example in this case for 32 nodes the communication overhead is around 25%.

One thing we can do to improve communication is to make sure that we are using the full compute capabilities of our GPU. Because Pytorch is optimizing the overlap between communication and compute, increasing the compute performed between communication will lead to better throughput. In the following plot we increased the local batch size from 64 to 128 and we can see the scaling efficiency increased to around 89% for 32 nodes.

<img src="tutorial_images/scale_perfEff_bs128.png" width="500">

Also to understand better the reason for this improvement we can look at the following plot of the communication overhead. The blue lines are with batch size of 128 and the orange lines with batch size 64. The difference between the solid and dotted lines is smaller for larger batch size as expected. For example for 32 nodes we see an improvement in the communication overhead from 25% for batch size 64, to 12% for batch size 128.

<img src="tutorial_images/scale_perfDiffBS.png" width="500">

### Profiling with Nsight Systems

Using the optimized options for compute and I/O, we profile the communication baseline with 
4 GPUs (1 node) on Perlmutter: 
```
ENABLE_PROFILING=1 PROFILE_OUTPUT=4gpu_baseline sbatch -n 4 ./submit_pm.sh --config=bs64_opt --num_epochs 4 --num_data_workers 8 --local_batch_size 16 --enable_manual_profiling
```
Considering both the case of strong scaling and large-batch training limitation, the 
`local_batch_size`, i.e. per GPU batch size, is set to 16 to show the effect of communication. Loading this profile ([`4gpu_baseline.qdrep`](sample_nsys_profiles/4gpu_baseline.qdrep)) in Nsight Systems will look like this: 
![NSYS 4gpu_Baseline](tutorial_images/nsys_4gpu_baseline.png)
where the stream 20 shows the NCCL communication calls. 

By default, for our model there are 8 NCCL calls per iteration, as shown in zoomed-in view:
![NSYS 4gpu_Baseline_zoomed](tutorial_images/nsys_4gpu_baseline_zoomed.png)

The performance of this run:
```
2021-11-10 04:03:37,792 - root - INFO - Time taken for epoch 2 is 61.7418851852417 sec, avg 1061.4512304471264 samples/sec
2021-11-10 04:03:37,792 - root - INFO -   Avg train loss=0.006371
2021-11-10 04:03:41,047 - root - INFO -   Avg val loss=0.006337
2021-11-10 04:03:41,048 - root - INFO -   Total validation time: 3.254544973373413 sec
2021-11-10 04:04:32,869 - root - INFO - Time taken for epoch 3 is 51.81808805465698 sec, avg 1264.7321130581577 samples/sec
2021-11-10 04:04:32,869 - root - INFO -   Avg train loss=0.005793
2021-11-10 04:04:36,134 - root - INFO -   Avg val loss=0.005889
2021-11-10 04:04:36,134 - root - INFO -   Total validation time: 3.2647454738616943 sec
2021-11-10 04:05:27,672 - root - INFO - Time taken for epoch 4 is 51.53450584411621 sec, avg 1271.6916350810875 samples/sec
2021-11-10 04:05:27,672 - root - INFO -   Avg train loss=0.005587
2021-11-10 04:05:30,891 - root - INFO -   Avg val loss=0.005936
2021-11-10 04:05:30,891 - root - INFO -   Total validation time: 3.2182624340057373 sec
```

### Adjusting DistributedDataParallel options

The [tuning knobs](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
for `DistributedDataParallel` includes `broadcast_buffers`, `bucket_cap_mb`, etc. `broadcast_buffers` adds 
additional communication (syncing buffers) and is enabled by default, which is often not necessary. `bucket_cap_mb` 
sets a upper limit for the messsage size per NCCL call, adjusting which can change the total number of communication 
calls per iteration. The proper bucket size depends on the overlap between communication and computation, and requires 
tunning. 

Since there is no batch norm layer in our model, it's safe to disable the `broadcast_buffers` with the added knob `--disable_broadcast_buffers`:
```
ENABLE_PROFILING=1 PROFILE_OUTPUT=4gpu_nobroadcast sbatch -n 4 ./submit_pm.sh --config=bs64_opt --num_epochs 4 --num_data_workers 8 --local_batch_size 16 --enable_manual_profiling --disable_broadcast_buffers
```
Loading this profile ([`4gpu_nobroadcast.qdrep`](sample_nsys_profiles/4gpu_nobroadcast.qdrep)) in Nsight Systems will look like this:
![NSYS 4gpu_nobroadcast](tutorial_images/nsys_4gpu_nobroadcast.png)
The per step timing is slightly improved comparing to the baseline. 

The performance of this run: 
```
2021-11-10 04:12:07,932 - root - INFO - Time taken for epoch 2 is 62.6831419467926 sec, avg 1045.5123652804289 samples/sec
2021-11-10 04:12:07,932 - root - INFO -   Avg train loss=0.006372
2021-11-10 04:12:11,173 - root - INFO -   Avg val loss=0.006370
2021-11-10 04:12:11,173 - root - INFO -   Total validation time: 3.2399580478668213 sec
2021-11-10 04:13:01,406 - root - INFO - Time taken for epoch 3 is 50.23048114776611 sec, avg 1304.705798202663 samples/sec
2021-11-10 04:13:01,406 - root - INFO -   Avg train loss=0.005815
2021-11-10 04:13:04,636 - root - INFO -   Avg val loss=0.005902
2021-11-10 04:13:04,636 - root - INFO -   Total validation time: 3.22876238822937 sec
2021-11-10 04:13:54,472 - root - INFO - Time taken for epoch 4 is 49.83389210700989 sec, avg 1315.088933035222 samples/sec
2021-11-10 04:13:54,473 - root - INFO -   Avg train loss=0.005614
2021-11-10 04:13:57,722 - root - INFO -   Avg val loss=0.005941
2021-11-10 04:13:57,723 - root - INFO -   Total validation time: 3.2491915225982666 sec
```
Comparing to the baseline, there are few percentages (performance may slightly vary run by run) improvement in `samples/sec`. 

To show the effect of the message bucket size, we add another knob to the code, `--bucket_cap_mb`. The current 
default value in PyTorch is 25 mb. We profile a run with 100 mb bucket size with following command:
```
ENABLE_PROFILING=1 PROFILE_OUTPUT=4gpu_bucket100mb sbatch -n 4 ./submit_pm.sh --config=bs64_opt --num_epochs 4 --num_data_workers 8 --local_batch_size 16 --enable_manual_profiling --disable_broadcast_buffers --bucket_cap_mb 100
```
Loading this profile ([`4gpu_bucketcap100mb.qdrep`](sample_nsys_profiles/4gpu_bucketcap100mb.qdrep)) in Nsight Systems (zoomed in to a single iteration) will look like this:
![NSYS 4gpu_bucketcap100mb_zoomed](tutorial_images/nsys_4gpu_bucketcap100mb_zoomed.png)
the total number of NCCL calls per step now reduced to 5. 

The performance of this run:
```
2021-11-10 04:19:48,472 - root - INFO - Time taken for epoch 2 is 59.066428899765015 sec, avg 1109.5304256706254 samples/sec
2021-11-10 04:19:48,472 - root - INFO -   Avg train loss=0.006478
2021-11-10 04:19:51,711 - root - INFO -   Avg val loss=0.006588
2021-11-10 04:19:51,712 - root - INFO -   Total validation time: 3.239215612411499 sec
2021-11-10 04:20:41,475 - root - INFO - Time taken for epoch 3 is 49.75986886024475 sec, avg 1317.0452716437817 samples/sec
2021-11-10 04:20:41,475 - root - INFO -   Avg train loss=0.005917
2021-11-10 04:20:44,730 - root - INFO -   Avg val loss=0.006044
2021-11-10 04:20:44,730 - root - INFO -   Total validation time: 3.2542178630828857 sec
2021-11-10 04:21:34,517 - root - INFO - Time taken for epoch 4 is 49.78394103050232 sec, avg 1316.4084370067546 samples/sec
2021-11-10 04:21:34,517 - root - INFO -   Avg train loss=0.005700
2021-11-10 04:21:37,772 - root - INFO -   Avg val loss=0.006073
2021-11-10 04:21:37,773 - root - INFO -   Total validation time: 3.2548396587371826 sec
```
Similarly, to understand the cross node performance, we run the baseline and optimized options with 2 nodes on Perlmutter. 

Baseline:
```
ENABLE_PROFILING=1 PROFILE_OUTPUT=8gpu_baseline sbatch -N 2 ./submit_pm.sh --config=bs64_opt --num_epochs 4 --num_data_workers 8 --local_batch_size 16 --enable_manual_profiling 
```
and the performance of the run: 
```
2021-11-10 02:41:30,680 - root - INFO - Time taken for epoch 2 is 44.45261096954346 sec, avg 1474.2891040731388 samples/sec
2021-11-10 02:41:30,710 - root - INFO -   Avg train loss=0.007586
2021-11-10 02:41:32,457 - root - INFO -   Avg val loss=0.007256
2021-11-10 02:41:32,457 - root - INFO -   Total validation time: 1.7458698749542236 sec
2021-11-10 02:42:08,002 - root - INFO - Time taken for epoch 3 is 35.54009485244751 sec, avg 1844.0018315113414 samples/sec
2021-11-10 02:42:08,028 - root - INFO -   Avg train loss=0.006422
2021-11-10 02:42:09,688 - root - INFO -   Avg val loss=0.006547
2021-11-10 02:42:09,688 - root - INFO -   Total validation time: 1.6595783233642578 sec
2021-11-10 02:42:45,635 - root - INFO - Time taken for epoch 4 is 35.94469451904297 sec, avg 1823.245429594067 samples/sec
2021-11-10 02:42:45,644 - root - INFO -   Avg train loss=0.006166
2021-11-10 02:42:47,310 - root - INFO -   Avg val loss=0.006547
2021-11-10 02:42:47,310 - root - INFO -   Total validation time: 1.6650199890136719 sec
```
Optimized:
```
ENABLE_PROFILING=1 PROFILE_OUTPUT=8gpu_bucket100mb sbatch -N 2 ./submit_pm.sh --config=bs64_opt --num_epochs 4 --num_data_workers 8 --local_batch_size 16 --enable_manual_profiling --disable_broadcast_buffers --bucket_cap_mb 100
```
and the performance of the run:
```
2021-11-10 02:41:28,509 - root - INFO - Time taken for epoch 2 is 43.84619975090027 sec, avg 1494.67913689953 samples/sec
2021-11-10 02:41:28,528 - root - INFO -   Avg train loss=0.007528
2021-11-10 02:41:30,271 - root - INFO -   Avg val loss=0.007238
2021-11-10 02:41:30,272 - root - INFO -   Total validation time: 1.742598056793213 sec
2021-11-10 02:42:05,129 - root - INFO - Time taken for epoch 3 is 34.85356664657593 sec, avg 1880.3240616534827 samples/sec
2021-11-10 02:42:05,136 - root - INFO -   Avg train loss=0.006444
2021-11-10 02:42:06,803 - root - INFO -   Avg val loss=0.006532
2021-11-10 02:42:06,804 - root - INFO -   Total validation time: 1.6663029193878174 sec
2021-11-10 02:42:42,100 - root - INFO - Time taken for epoch 4 is 35.293962717056274 sec, avg 1856.8614843673777 samples/sec
2021-11-10 02:42:42,123 - root - INFO -   Avg train loss=0.006195
2021-11-10 02:42:43,763 - root - INFO -   Avg val loss=0.006568
2021-11-10 02:42:43,786 - root - INFO -   Total validation time: 1.6387364864349365 sec
```
Note that the batch size is set to a small value to tune the knobs at smaller scale. To have a better scaliing efficiency, we
 want to increase the per GPU compute intensity by increasing the per GPU batch size. 

## Putting it all together

With all of our multi-GPU settings and optimizations in place, we now leave it to you to take what you've learned and try to achieve the best performance on this problem. Specifically, try to further tune things to either reach the lowest possible validation loss, or converge to the single-GPU validation loss (`~4.7e-3`) in the shortest amount of time. Some ideas for things to adjust are:
* Further tune `num_epochs` to adjust how long it takes for learning rate to decay, and for training to conclude.
* Play with the learning rate: try out a different scaling rule, such as linear scale-up of learning rate, or come up with your own learning rate schedule.
* Change other components, such as the optimizer used. Here we have used the standard Adam optimizer, but many practitioners also use the SGD optimizer (with momentum) in distributed training.

The [PyTorch docs](https://pytorch.org/docs/stable/index.html) will be helpful if you are attempting more advanced changes.
