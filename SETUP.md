# Setup Instructions for JayQuery
> These instructions are meant to set up a local environment up to the most recent iterations of the project. This does not necessarily mean that the project with be completely functional after completing these setup instructions.

## Basic setup for the repository and environment
```bash
# 1. Clone and set up environment
git clone https://github.com/Peilong/JayQuery.git
cd JayQuery
conda create -n jayquery python=3.11 && conda activate jayquery
pip install -r requirements.txt
pip install -e .

# 2. Copy and configure environment variables
cp .env.example .env # fill in WANDB_API_KEY

# 3. Download OULAD dataset
bash scripts/download_oulad.sh
```

## Testing the flwr server/client
```bash
#Set flwr to simulation mode with 2 virtual SuperNodes.
flwr federation simulation-config --num-supernodes=2

#Run the flwr server/client in a simulation.
flwr run .

#This doesn't show much, but it is functional for starting the flwr environment.
```

## Run Model Baselines
```bash
#Wandb login, enter the API key you saved to the env variables if asked.
wandb login

#Can replace the --run all with --run x, where x is the model you want to run. The model names are defined in the baselines/run_baselines file.
#Also can replace any model parameters in exp01 with any that you want the baselines to use.
python -m baselines.run_baselines --config experiments/configs/exp01_centralized_baselines.yaml --run all

```