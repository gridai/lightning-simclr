import torchvision.transforms as transforms

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, Tuple


class SimCLRTrainTransform(object):

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.,
        normalize: Optional[Callable[[Tensor], Tensor]] = None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5)
            )

        data_transforms = transforms.Compose(data_transforms)

        self.online_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.simclr_transform = transforms.Compose([data_transforms, self.final_transform])
        self.online_train_transform = transforms.Compose([self.online_train_transform, self.final_transform])

    def __call__(self, sample: Tensor) -> Tuple[Tensor, Tensor]:
        transform = self.simclr_transform

        view1 = transform(sample)
        view2 = transform(sample)
        online_view = self.online_train_transform(sample)

        return view1, view2, online_view


class SimCLREvalTransform(object):

    def __init__(
        self,
        input_height: int = 224,
        resize_height: Optional[Union[int, str]] = 'default',
        gaussian_blur: bool = True,
        jitter_strength: float = 1.,
        normalize: Optional[Callable[[Tensor], Tensor]] = None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        if resize_height is None:
            self.resize_height = self.input_height
        elif resize_height == 'default':
            self.resize_height = int(self.input_height + 0.1 * self.input_height)
        else:
            assert isinstance(resize_height, int)
            self.resize_height = resize_height

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5)
            )

        data_transforms = transforms.Compose(data_transforms)

        self.online_eval_transform = transforms.Compose([
            transforms.Resize(self.resize_height),
            transforms.CenterCrop(self.input_height),
        ])

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.simclr_transform = transforms.Compose([data_transforms, self.final_transform])
        self.online_eval_transform = transforms.Compose([self.online_eval_transform, self.final_transform])

    def __call__(self, sample: Tensor) -> Tuple[Tensor, Tensor]:
        transform = self.simclr_transform

        view1 = transform(sample)
        view2 = transform(sample)
        online_view = self.online_eval_transform(sample)

        return view1, view2, online_view
