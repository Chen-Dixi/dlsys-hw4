import numpy as np

# 数据在转为needle的 NDArray 之前，通过numpy来存储。后面再转为 tensor。

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: ... x H x W x C NDArray of an image, where ... means an arbitrary number of leading
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            # 水平翻转  [..., H, W] shape, where ... means an arbitrary number of leading
            return np.flip(img, -2)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img:... x H x W x C NDArray of an image, where ... means an arbitrary number of leading
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        # 填充0
        img_pad = np.zeros_like(img)
        H, W = img.shape[-3], img.shape[-2]
        if abs(shift_x) >= H or abs(shift_y) >= W:
            return img_pad
        
        img_start_x, img_end_x = max(0, shift_x), min(H, H + shift_x)
        img_start_y, img_end_y = max(0, shift_y), min(W, W + shift_y)
        pad_start_x, pad_end_x = max(0, -shift_x), min(H, H - shift_x)
        pad_start_y, pad_end_y = max(0, -shift_y), min(W, W - shift_y)
        img_pad[:, pad_start_x: pad_end_x, pad_start_y: pad_end_y, :] = img[:, img_start_x: img_end_x, img_start_y: img_end_y, :]
        return img_pad
        ### END YOUR SOLUTION
