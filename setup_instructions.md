# To install robogym:
docker pull rodrigodelazcano/ray-mujoco:latest-gpu

# example command to start the docker image running in the background:
docker run --gpus all -d --user 0 --rm --name collect_data \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro -v /home/${USER}:/home/${USER}:ro \
  -v /home/${USER}/Documents/docker_share:/share \
  -it rodrigodelazcano/ray-mujoco:latest-gpu /bin/bash

# set up implicit-iterative-inference when inside the docker image:
conda create --name roboimp --clone base

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

cd implicit-iterative-inference

pip install -r requirements.txt