import math
from argparse import ArgumentParser
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from resnet import resnet18, resnet50
from online_eval import OnlineFineTuner
from cifar10 import CIFAR10DataModule, cifar10_normalization
from transforms import SimCLRTrainTransform, SimCLREvalTransform


def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """
    Linear warmup for warmup_steps, optionally with cosine annealing or
    linear decay to 0 at total_steps
    """
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        arch: str = 'resnet50',
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        super().__init__()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim

        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.encoder = self.init_model()

        self.projection = Projection(
            input_dim=self.hidden_mlp,
            hidden_dim=self.hidden_mlp,
            output_dim=self.feat_dim
        )

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    def init_model(self):
        if self.arch == 'resnet18':
            backbone = resnet18
        elif self.arch == 'resnet50':
            backbone = resnet50

        return backbone(first_conv3x3=True, remove_first_maxpool=True)

    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)

    def shared_step(self, batch):
        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        # linear warmup with optional cosine decay
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--fp32", action='store_true')

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


def cli_main():
    parser = ArgumentParser()

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    val_split = 5000
    if args.num_nodes * args.gpus * args.batch_size > val_split:
        val_split = args.num_nodes * args.gpus * args.batch_size

    dm = CIFAR10DataModule(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
    )

    args.num_samples = dm.num_samples

    args.input_height = dm.size()[-1]
    args.temperature = 0.5

    normalization = cifar10_normalization()

    args.gaussian_blur = False
    args.jitter_strength = 0.5

    dm.train_transforms = SimCLRTrainTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalTransform(
        input_height=args.input_height,
        resize_height=None,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = SimCLR(**args.__dict__)

    if args.arch == 'resnet18':
        encoder_output_dim = 512
    else:
        encoder_output_dim = 2048

    online_evaluator = OnlineFineTuner(encoder_output_dim=encoder_output_dim, num_classes=dm.num_classes)

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint, online_evaluator]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
