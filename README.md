# MARL-DMFB
A cooperative multi-agent reinforcement learning framework for droplet routing for droplet routing in DMFB

## Log 
- Aug 29
    - scan through and understand the code
    - trial run

- Sept 3
    - upload code
    - make to-do list
    - try env.render
    - add line 200-211 in dmfb.py
    - hardcode the dimension of the model for fov=5 for RCNN model

## To-do list
- [x] animate the environment
- [x] spawn droplets at the edge of the chip
- [ ] implement CRNN_Attention

## Train model
python main.py --chip_size=10 --drop_num=4 

training data will be saved in "TrainResult/vdn/10by10-4d0b/"

trained model will be saved in "model/vdn/10by10-4d0b/"

## Evaluate model in health mode
python main.py --evaluate --load_model --chip_size=50 --drop_num=4 --evaluate_epoch=100

This will evaluate the performance of the model: "model/vdn/50by50-4d0b/rnn_net_params.pkl" and "model/vdn/50by50-4d0b/vdn_net_params.pkl"

## Evaluate model with electrodes degrade
python evaDegre.py --chip_size=10 --drop_num=4 --evaluate_epoch=40

This will evaluate the performance of the model: "model/vdn/10by10-4d0b/rnn_net_params.pkl" and "model/vdn/10by10-4d0b/vdn_net_params.pkl"

The data will be saved in "DgreData/10by10-4d0b"

## Parameters
You can find more usages or change the parameters of the algorithm in the file "common/arguments.py"
