
# 목적 
#  스크립트별 설명
- mae_pretrain: domain adaption with small learning_rate.
- mae_finetune: finetune above model
- make_dataset: deprecated
- tf_mae: tensorflow keras-io에서 제공하는 masked autoencoder 모델을 수정하여 테스트.  가지고 시험함. size 32로 하기위해 pretrain이 안된 작은 모델을 새롭게 구성해서 학습하나 학습이 제대로 되지 않음.


# experements
- triplet loss를 통해 train folder를 학습한 후, unlabeled test set에 pueod label을 만들어 supervised contrasive loss를 적용하려고 함. 상품의 경우 rotation이 제한되고 광량도 일정하므로 일반적인 ssl에 적용하는 augmentation들이 제대로 먹히지 않을 것이라고 생각함. 이것 전에 clustering이 제대로 먹히나 아래 방법을 먼저 적용

- tf_siamese network: 제대로 학습되지 않음. 어차피 test dir 를 활용하기 위해서는 어느식으로든 ssl 개념이 들어가야함. augmentation 영향을 거의 받지 않는 masked autoencoder를 시험하기로 하고 다음으로 넘어감

- tf_mae:  가지고 시험함. size 32로 하기위해 pretrain이 안된 작은 모델을 새롭게 구성해서 학습하나 학습이 제대로 되지 않음. pretrained된 모델이 있는 huggingface로 넘어감

- hf_mae: 먼저 imagenet pretrained된 model을 불러와서 낮은 lr(1e-5)로 pretraining을 더함. domain adaptation 목적임. 
그 후, train dir 이미지를 불러와서 8:2로 분할하여 finetuning 함
그 후 unlabeled 된 test dir 성능을 평가하니 acc at 1이 50% 정도 나옴.

# metric 연구




# codespace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /tmp/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -y -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda install python=3.8 pip -y
pip install -r requirements.txt
# download data.tar.gz
tar -xf data.tar.gz

