# Grid Demo | Self-supervised Learning with Lightning
In this demo example, you'll train a self-supervised model using Grid.

If you haven't already set up the Grid CLI, follow this [1 minute guide](https://app.gitbook.com/@grid-ai/s/grid-cli/start-here/typical-workflow-cli-user#step-0-install-the-grid-cli) on how to install the Grid CLI.

**TLDR:** 
`pip install lightning-grid --upgrade`

`grid login`

## Training Parameters
Here are the parameters we'll specify to `grid train`:

**Grid flags:**
1. **--grid_name:** the name to use for the Grid training run
2. **--grid_instance_type:** defines number of GPUs and memory
3. **--grid_gpus:** the number of GPUs per experiment

Then we'll specify the script we're using to train our model followed by the script arguments. 

**Script:** `src/train.py`

These are the arguments defined by the `train.py` script:

**Script arguments:**
1. batch_size
2. num_workers
3. exclude_bn_bias
4. max_epochs
5. gpus

Notice there are two gpu arguments - one that gets passed to `grid train` and another which gets passed to the script. The `--grid_gpus` flag lets grid know how many GPUs should be allocated to each experiment while the `--gpus` flag tells the Lightning Trainer to use the allocated number of GPUs to run the experiment. At this time both parameters are needed and should be set to the same value. 

## Run on a single GPU:
Using the parameters above, we define a training run called 'simclr-baseline' with a single GPU. Submit the command below to create the training run in Grid. 

```
grid train --grid_name simclr-baseline \
    --grid_instance_type p3.2xlarge \
    --grid_gpus 1 \
    src/train.py \
    --batch_size 256 \
    --num_workers 16 \
    --exclude_bn_bias \
    --max_epochs 800
```
**Run Status**
After submitting the run, you can check the status in the CLI by running `grid status simclr-baseline` OR you can check the status in the Web UI by running `grid view simclr-baseline`. 

## Run with 8 V100s:

To submit the same run as above, but with 8 V100s instead, update the `grid_instance_type`, `--grid_gpus`, and `--gpus` flag to reflect the new resource requirements. 

```
grid train --grid_name simclr-baseline \
    --grid_instance_type p3.16xlarge \
    --grid_gpus 8 \
    src/train.py \
    --gpus 8 \
    --batch_size 256 \
    --num_workers 16 \
    --exclude_bn_bias \
    --max_epochs 800
```
