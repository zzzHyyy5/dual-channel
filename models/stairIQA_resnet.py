import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

class StaircaseNetwork(nn.Module):
    def __init__(self):
        super(StaircaseNetwork, self).__init__()
        
        # 使用预训练的MobileNetV2特征提取器
        mobilenet_features = nn.Sequential(*list(models.mobilenet_v2(weights='DEFAULT').children())[0])

        self.feature_extraction_stem = nn.Sequential()
        self.feature_extraction1 = nn.Sequential()
        self.feature_extraction2 = nn.Sequential()
        self.feature_extraction3 = nn.Sequential()
        self.feature_extraction4 = nn.Sequential()

        # 将MobileNetV2分成多个层级
        for x in range(0, 4):
            self.feature_extraction_stem.add_module(str(x), mobilenet_features[x])

        for x in range(4, 7):
            self.feature_extraction1.add_module(str(x), mobilenet_features[x])

        for x in range(7, 11):
            self.feature_extraction2.add_module(str(x), mobilenet_features[x])

        for x in range(11, 17):
            self.feature_extraction3.add_module(str(x), mobilenet_features[x])

        for x in range(17, 19):
            self.feature_extraction4.add_module(str(x), mobilenet_features[x])

        # 定义超结构（hyper-structures）
        self.hyper1_1 = self.hyper_structure(24, 32)
        self.hyper2_1 = self.hyper_structure(32, 64)
        self.hyper3_1 = self.hyper_structure(64, 160)
        self.hyper4_1 = self.hyper_structure(160, 1280)

        self.quality = self.quality_regression(1280, 128, 1)

    def hyper_structure(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def quality_regression(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels)
        )

    def forward(self, x):
        x = self.feature_extraction_stem(x)

        x_hyper1 = self.hyper1_1(x)
        x = self.feature_extraction1(x)

        x_hyper1 = self.hyper2_1(x_hyper1 + x)
        x = self.feature_extraction2(x)

        x_hyper1 = self.hyper3_1(x_hyper1 + x)
        x = self.feature_extraction3(x)

        x_hyper1 = self.hyper4_1(x_hyper1 + x)
        x = self.feature_extraction4(x)

        x = x + x_hyper1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.quality(x)

        return x

class SelfCompetitorModel(nn.Module):
    def __init__(self, original_model):
        super(SelfCompetitorModel, self).__init__()
        self.model = original_model
        
    def prune(self, pruning_ratio=0.7):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                with torch.no_grad():
                    mask = param.abs() > torch.topk(param.abs().flatten(), int(pruning_ratio * param.numel())).values[-1]
                    param.data *= mask.view(param.size())

    def forward(self, x):
        return self.model(x)

class HardExampleMiningLoss(nn.Module):
    def __init__(self):
        super(HardExampleMiningLoss, self).__init__()

    def forward(self, full_model_output, pruned_model_output, difficulty_factor=2.0):
        # 计算困难样本挖掘损失，基于全目标模型和自竞争者模型的输出差异
        mse_loss = F.mse_loss(full_model_output, pruned_model_output)
        
        # 在困难样本上施加更多的关注，通过增加损失值
        return mse_loss * difficulty_factor

class IterativeMixedDatabaseTraining(nn.Module):
    def __init__(self, model, pruning_ratio=0.7):
        super(IterativeMixedDatabaseTraining, self).__init__()
        self.model = model
        self.self_competitor = SelfCompetitorModel(model)
        self.pruning_ratio = pruning_ratio
        self.hard_example_loss = HardExampleMiningLoss()

    def forward(self, x):
        # 原始模型的输出
        full_output = self.model(x)
        
        # 剪枝操作，生成自竞争者模型
        self.self_competitor.prune(self.pruning_ratio)
        pruned_output = self.self_competitor(x)
        
        # 计算困难样本挖掘损失，应用困难样本的加权损失
        loss = self.hard_example_loss(full_output, pruned_output, difficulty_factor=2.0)

        # 返回全目标模型输出和损失
        return full_output, loss

# Sample usage:
if __name__ == '__main__':
    model = StaircaseNetwork()
    iterative_model = IterativeMixedDatabaseTraining(model)
    x = torch.randn(16, 3, 224, 224)  # Example input tensor
    output, loss = iterative_model(x)
    print(f"Output shape: {output.shape}, Loss: {loss.item()}")
