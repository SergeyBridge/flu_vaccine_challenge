#!/bin/bash          

if [ -n "$1" ]
then

# JUPYTER_NOTEBOOK_CONFIG="/home/sergey/mnt/st1500/Usr/Sergey/TheJob/clientsProjectsNew/wvproject/ycloud/gpu/jupyter_notebook_config.py"
JUPYTER_NOTEBOOK_CONFIG="/home/sergey/mnt/st1500/Usr/Sergey/TheJob/Challenges/riiidNew/ycloud/jupyter_notebook_config.py"

echo "*****************************"
echo "*                           *"
echo "*    local configurations   *"
echo "*                           *"
echo "*****************************"
rm ~/.ssh/known_hosts
echo   git credentials ----
scp ~/.ssh/gitHubSergeyBridge_rsa $1:~/.ssh
scp /etc/locale.gen $1:
ssh $1 sudo mv locale.gen /etc/
scp  ./apt_install_packages.txt $1:
scp  ./pip_requirements.txt $1:

ssh $1 mkdir ".jupyter"
scp  ./jupyter_notebook_config.py $1:.jupyter/jupyter_notebook_config.py

echo "*****************************"
echo "*                           *"
echo "*    remote configurations  *"
echo "*                           *"
echo "*****************************"

ssh $1 -tt << EOF
  # *****************  locale settings  **************
    sudo localedef ru_RU.UTF-8 -i ru_RU -f UTF-8
    sudo localectl set-locale LANG=en_US.UTF-8
    sudo localectl set-locale LANGUAGE=en_US.UTF-8
    sudo locale-gen
    sudo locale
  # *****************  apt update  **************
    sudo apt-get update
  # ------------------------sudo apt-get --yes upgrade------------------------
    sudo apt-get --yes upgrade
  # PRESS CTRL-C
EOF
ssh $1 -tt << EOF
  # ****************** apt_install_packages.txt packages installing *******************************
    xargs -a <(awk '! /^ *(#|$)/' "./apt_install_packages.txt") -r -- sudo apt-get install --yes
  # PRESS CTRL-C
EOF
ssh $1 -tt << EOF
  # ====== anaconda installing ======================
    wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
    bash  ./Anaconda3-2020.02-Linux-x86_64.sh -b -p ~/anaconda3
  # PRESS CTRL-C
EOF
ssh $1 -tt << EOF
  # ***********  conda init *********************************
    ~/anaconda3/bin/conda init
    source ~/.bashrc
  # ====== pandas installing ======================
    conda config --add channels conda-forge
    conda update -n base --yes -c defaults conda
  # PRESS CTRL-C
EOF
ssh $1 -tt << EOF
    conda install --yes jupyter
    conda activate
    pip install  -r pip_requirements.txt
  # PRESS CTRL-C
EOF
ssh $1 -tt << EOF
  # ========== git cloning ===========================
    git config --global user.name "SergeyBridge"
    git config --global user.email "sergey.novozhilov@yandex.ru"
    GIT_SSH_COMMAND='ssh -i ~/.ssh/gitHubSergeyBridge_rsa \
      -o UserKnownHostsFile=/dev/null \
      -o StrictHostKeyChecking=no' \
      git clone git@github.com:SergeyBridge/flu_vaccine_challenge.git
    cd ~/flu_vaccine_challenge/drivendata_flu_shot
    git init
    jupyter lab

    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/gitHubSergeyBridge_rsa
  # PRESS CTRL-C
EOF
else
  echo "gpu.bat gpu@host_address: expected 1 parameter - remote host address "
fi
