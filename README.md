Propose a system to ensure robust reinforcement learning using decision transformer 
===============================

## About 

### Requirements
- Python >= 3.9 
- Conda 

### How-to
Clone this repository and run: 

```bash
conda env create -f environment.yaml
```
and activate the environment with: 
```bash 
conda activate hanh_env
```

### Training the model 
1. First, download the dataset: 
```bash
cd mujoco && pip install -r requirements.txt
```
and running: 
```bash 
python download_dataset.py
```

2. Training with fixable configurations: 

```bash
python main.py
```





