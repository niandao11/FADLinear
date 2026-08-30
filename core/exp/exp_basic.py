import torch
import torch.nn as nn

class Exp_Basic(object):

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self) -> torch.device:
        if self.args.use_gpu and torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name()}')
            return torch.device('cuda')
        print('Using CPU')
        return torch.device('cpu')

    def _build_model(self):
        raise NotImplementedError("子类必须实现模型构建")

    def _get_data(self, flag: str) -> tuple:
        raise NotImplementedError("子类必须实现数据加载")

    def _select_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError("子类必须实现优化器选择")

    def _select_criterion(self) -> torch.nn.Module:
        raise NotImplementedError("子类必须实现损失函数选择")

    def train(self, setting: str) -> nn.Module:
        raise NotImplementedError("子类必须实现训练逻辑")

    def vali(self, vali_data, vali_loader, criterion) -> float:
        raise NotImplementedError("子类必须实现验证逻辑")

    def test(self, setting: str, test: int = 0) -> None:
        raise NotImplementedError("子类必须实现测试逻辑")

    def predict(self, setting: str, load: bool = False):
        raise NotImplementedError("子类必须实现预测逻辑")
