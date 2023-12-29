# AIb_Project

## 12.29
### 进展
增加数据集预处理代码，见dataset_pre.py。更新模型微调模板，调用函数在finetune_model.py中。目前已完成模型在ChestX数据集上的微调，代码见model_policy_experiment.ipynb.

### TODO
针对模型结构改进的微调实验：

- 更换数据集，照着现有代码改
- 调参(目前关注的是learnning rate, epoch_num)，不同方法可能需要不同的参数以达到更好的效果
- 最终结果记录与结果展示(表格 or 作图)
- 更多关注点 IDEAS？微调过程？...

数据集本身带来的问题：

- 不平衡：更换评价指标？
- 不同数据集类别数量不一样：采取指标增长率(i.e.相对评价方法), 而不是绝对评价方法。以保证相对的公平。同时存在隐患的，chestX是二分类，如果用flowers102的话那就是102个类，如果flowers调出来还没有chestX好的话考虑换掉flowers(这个再说)

baseline:
- Random
- ProtoNet(尝试复用github代码)

### WARNING
- 每一次微调都在进一步利用内存，大约6-7次微调(载入新模型并训练)之后会爆内存。无法一次完成同一个数据集的所有微调，所以最终结果（暂时用accuracy）需手动记录，最后画图或直接填入表格。
- 请注意数据集存放格式! 目前data下存放原数据Coronahack-Chest-XRay-Dataset，预处理后存在chestX中；chestX中有原始数据的train和test,这两个文件夹里面又分别有两个文件夹存放两个类的数据；5_shot_dataset中是从train抽出的数据，每个类抽五张
- VGG16结构如下(我们微调的时候修改的都是classifier)

VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=2, bias=True)
  )
)
