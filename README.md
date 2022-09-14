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

- Sept 7
    - save the result of different base net in different folder
    - implement a simple attention model
    - tested crnn and crnn_attention model
        - result: 
        - chip_size=6, droplet=2: similar and ok performance (>95% success rate)
        - chip_size=8, droplet=3: both have poor performance (<70% success rate)
        - evaluation:
        - suspect 1. the need to refine the code for fov=5 
            - because too few data is provided to make dicision
            - tried chip_size=10, droplet=3: not being able to duplicate the paper result
        - suspect 2. the need to increase the max step allowed
            - because all droplets starts at the edge

- Sept 9
    - aim: attain similar performance for fov5 compared with fov9 (~95%)
    - improve cnn code for fov=5 by 
        - adding padding=1 (~80% success rate)
        - kernel size=2 (~85% success rate)
        - kernal size=2 and 64 output for conv2 (<85% success rate)
        - kernel size=2 and add a fc2 layer (<85% success rate)
        - using attention:
        - use attention layer before rnn layer
        - using Inception:
        - using inception-like model and kernel size=3 (<85% success rate)
        - using inception-like model and kernel size=2 (~80% success rate)
        - add one layer before rnn for inception cuz too much node (<80%)
        - remove path one from inception (<85%)
        - widen the cnn by including (1,5) and (5,1) kernel size (~85%)

- Sept 12
    - increase the rnn_hidden_dim back to 128
    - improve cnn code for fov=5 by
        - widen the cnn by including (1,5) and (5,1) kernel size + 3 layer kernel size=2 (~86%)
    - cmd:
        - python main.py --n_steps=2500000 --chip_size=10 --drop_num=3 --net=crnn_inception --fov=5
        - python main.py --n_steps=2500000 --chip_size=10 --drop_num=3 --net=crnn_inception --fov=5 --load_model --load_model_name=0_ --evaluate --evaluate_epoch=1000
        


## To-do list
- [x] shoe the droplets environment
- [x] spawn droplets at the edge of the chip
- [ ] fov=5

## Train model
python main.py --chip_size=10 --drop_num=4 

training data will be saved in "TrainResult/vdn/crnn_9/10by10-4d0b/"

trained model will be saved in "model/vdn/crnn_9/10by10-4d0b/"

## Evaluate model in health mode
python main.py --evaluate --load_model --chip_size=50 --drop_num=4 --evaluate_epoch=100

This will evaluate the performance of the model: "model/vdn/crnn_9/50by50-4d0b/rnn_net_params.pkl" and "model/vdn/crnn_9/50by50-4d0b/vdn_net_params.pkl"

## Evaluate model with electrodes degrade
python evaDegre.py --chip_size=10 --drop_num=4 --evaluate_epoch=40

This will evaluate the performance of the model: "model/vdn/crnn_9/10by10-4d0b/rnn_net_params.pkl" and "model/vdn/crnn_9/10by10-4d0b/vdn_net_params.pkl"

The data will be saved in "DgreData/10by10-4d0b"

## Parameters
You can find more usages or change the parameters of the algorithm in the file "common/arguments.py"
