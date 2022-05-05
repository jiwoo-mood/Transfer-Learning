# Transfer-Learning(전이학습)
Transfer Learning for Computer Vision Tutorial
<!-- Line -->


## What is transfer learning
<!-- Line -->
cs231n leacture 7에서  ***CNNs을 학습하거나 이용할때 꼭 많은 데이터가 필요한가? 에 대해 물어본다. Transfer learning을 활용한다면 꼭 그렇지는 않다.***

<!--Image-->
![transferlearning](https://user-images.githubusercontent.com/84561436/166867877-f4e174c2-8c86-4f92-922f-04b75c506303.JPG)
<!--Image-->
![vp](https://user-images.githubusercontent.com/84561436/166871447-59fed288-7a82-4ec8-a861-e898cc5312f9.JPG)

아이디어는 간단하다. CNN Layer를 가지고 우선 ImageNet 과 같은 큰 데이터셋으로 학습을 시킨 후 우리가 적용하고자 하는 데이터셋에 적용을 시킨다. 적용하고자 하는 데이터셋이 작은 경우 일반적인 절차는 마지막의 FC Layer 을(최종 feature과 class scores간의 연결) 초기화시킨다. 방금 정의한 가중치 행렬은 초기화시키고 이전의 나머지 레이어들의 가중치는 freeze를 시킨다. 우리는 초기화한 레이어만 재사용을 하고 마지막 레이어만 가지고 데이터를 학습하게 된다. 적용하고자 하는 데이터가 많다면 learning rate을 줄여서 fine tuning을 할 수도 있다. 위의 표처럼 적용하고자 하는 모델의 데이터셋과 데이터 유사도에 따라 다르게 적용할 수 있다. 

규모가 매우 큰 모델을 학습시킬 때 처음부터 새로 학습시키는 것은 학습 속도가 느린 문제가 있다. 이러한 경우 **기존에 학습된 비슷한 모델이 있을 때 이 모델의 하위층(lower layer)을 가져와 재사용하는 것이 학습 속도를 빠르게 할 수 있을 뿐만 아니라 학습에 필요한 Training set도 훨씬 적다.**

전이학습은 **학습한 가중치의 일부를 능력이 유사하거나 새로운 분야의 신경망에 복사한 후, 그 상태로 재학습을 수행하는 것**을 의미한다. 전이 학습은 학습 데이터가 부족한 프로젝트를 진행하는 경우 매우 큰 도움이 된다. 전이 학습은 데이터의 수가 적을 때도(물론 많을 때도) 매우 효과적이며 학습 속도 또한 빠르고 전이 학습 없이 학습하는 것보다 훨씬 높은 정확도를 제공한다는 장점이 있다. 

예를 들어, 아래의 그림처럼 CIFAR10 데이터셋을 분류(비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭의 10개  클래스)하는 모델 A가 이 있다고 하자. 그런 다음, 분류된 CIFAR10 이미지에서 자동차의 종류를 분류하는 모델인 B를 학습시킨다고 할 때, 학습된 모델 A에서의 일부분(lower layer)을 재사용하여 모델 B를 학습 시킬 수 있다. 이러한 방법을 Transfer Learning이라고 한다.

<!--Image-->
![transferlearning2](https://user-images.githubusercontent.com/84561436/166870474-804ce5bf-7cb7-4da2-8479-dbacef5e3ed2.JPG)

참고할 수 있는 블로그 : 
* https://fabj.tistory.com/57 
* https://welcome-to-dewy-world.tistory.com/92
* https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
* https://lsjsj92.tistory.com/405



## Data
<!-- Line -->
Pytorch tutorials 에서 제공하는 데이터를 사용했습니다. 데이터를 불러오기 위해 torchvision과 torch.utils.data 패키지를 사용했습니다. 해당 데이터는 개미와 벌의 각각 train 이미지는 120개가 있고, 75개의 validation 이미지가 포함되어있습니다. 데이터는 [여기](https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html)에서 다운받을 수 있습니다. 


## Models
<!-- Line -->

``` python
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model
```

## Finetuning the convnet
<!-- Line -->
사전적으로 훈련된 모델을 로드하고 final fully connected layer를 리셋합니다. 우리는 초기화한 레이어만 재사용을 하고 마지막 레이어만 가지고 데이터를 학습하기 때문이죠. 

```Python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

## Result
<!-- Line -->
<!--Image-->
![결과](https://user-images.githubusercontent.com/84561436/166874667-eba803df-4ee0-4ef1-b37c-cac4121be72d.JPG)

사전적으로 훈련된 모델을 로드하고 레이어를 리셋한 후 재사용을 한 결과입니다. ants와 bees 을 구분하는데에 있어서 비슷한 모델을 일정 부분 freeze, 재사용해서 예측하게끔하는 방법이 Transfer Learning 입니다.
