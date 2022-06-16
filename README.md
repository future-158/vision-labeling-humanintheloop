# 목적 및 설명
- 상품image/상품label dataset으로 human in the loop pipeline 구성
- dataset은 private으로 열람이 제한되며,
상품 id2label은 ./dataset/data/catalog.csv,
이미지는 ./dataset/data/{train,test} 폴더에 나누어 있어야함

# 실행
1. make install # dependency installation
2. python exp/clip/finetune.py
를 실행하면 모델링 후 모델은 result/clip_finetune_human_v102_model_weight.pt에 저장되며, 
성능은 result/clip_finetune_human_v102_trial_result_0.csv에 저장됨
3. exp/clip/evalute.py를 실행하면 
위에서 저장한 모델을 불러와  test_dataset(375개)를 불러와서 성능을 계산하고
result/final_score.json에 저장함

1,2,3번 순서대로 실행한 후, result/final_score.json 파일을 확인하면 됨 

make reproduce 실행시 1,2,3번 순서대로 실행됨
# experements
- tf_mae: keras-io에서 제공하는 masked auto encoder 모델로 실험함. finetune 성능이 낮아 미리 학습된 weight이 없는 것이 원인이라고 판단하고 huggingface 아래 implementation으로 변경
- hf_mae: imagenet pretrained된 model을 불러와서 finetuning함. acc_1 성능은 50%로 낮게 나옴
- clip: zeroshot 성능이 top1 acc 50% 가량 나오며, 낮은 lr(1e-5)로 human in the loop 방식으로 finetuning 후 test set 성능이 top1 acc가 81.6%로 나옴

# metric
먼저 confident한 예측을 max probability가 THRESHOLD 이상인 예측으로 정의하고 아래 값들을 구함

TP_at_1 = confident & correct
TP_at_k = unconfident  & correct ( top k개 중에 true 라벨이 존재)
FP_at_k = unconfident & incorrect (top k개 중에 true 라벨이 없음)
FP_at_1 = confident & incorrect

FP_at_k의 경우 human in the loop 과정에서 정상 라벨로 변환되어 이후 train set에 포함되므로 TP_at_k 효과가 일부 있으며 이는 다음 epoch 이후에 나타나므로 이를 할인하여 미리 보완해 주어야함
이는 MAX_INPSECT_PER_ROUND / unlabel dataset에 비례하므로 max(1, 1000 / unlabel_size)를 곱하였으며,
이 값은 tp_at_k보다 클 수 없으므로 위에서 계산된 값에 0.1을 곱해줌

그렇게
TP_paradox = FP_at_k  * max(1, 1000 / unlabel_size) * 0.1
를 구한 후
TP = TP_at_1 + ( TP_at_k + TP_paradox) / k
FP = FP_at_k + FP_at_1
를 계산한 후,

TP / (TP + FP)를 최종 metric으로 사용함

# human in the loop 구성 방법
모델 예측값 MAX가 THRESHHOLD(0.5 사용)보다 작은 케이스들 중 MAX_INPSECT_PER_ROUND(1000개 사용)를 뽑아 라벨링하여 이를 학습자료에 계속 추가함
모델 예측값 MAX가 THRESHHOLD 큰 경우 바로 모델 예측값을 라벨로 사용하여 train set에 추가하였으나, 이는 일시적인 것으로 다음 epoch이 끝난 후 unlabel set로 돌아가 위의 과정을 반복하게 됨.
이는 human in the loop에서 사람 눈을 거친 경우 라벨이 확실하므로 한번 train set에 집어넣으면 끝이지만, 모델 예측값만 가지고 라벨링한 자료의 경우 초기 예측값이 부정확할 수 있으며 또한 모델이 가진 bias를 그대로 답습할 수 있으므로 이를 교정하기 위함임

# hyper parameter optimization
사용한 metric에서 human inspect time을 제대로 반영되지 않는 문제점을 개선하기 위해 multi objective hyper parameter optimization을 수행함.
latency와 modified_top_k_acc를 각각 minimization하는 하이퍼파라미터를 탐색


