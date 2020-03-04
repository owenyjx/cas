import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy
import scipy.misc
import os

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#load models
from torchvision.models import inception_v3
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd

#load attacks
from advertorch.attacks import LinfPGDAttack


model = inception_v3(pretrained=True)
model.eval()
model = model.to(device)

#load data from validation of imagenet
Batchsize = 1
correct_sum = torch.zeros(255)
transform = transforms.Compose([transforms.Resize(299),transforms.CenterCrop(299),transforms.ToTensor(),transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],std = [ 0.229, 0.224, 0.225 ]),])
transform2 = transforms.Compose([transforms.Resize(299),transforms.CenterCrop(299),transforms.ToTensor(),])
dataset = datasets.ImageFolder('/hd2/yangjiaxi/PDG_attack_imagenet/data/Imagenet/val',transform)
dataset2 = datasets.ImageFolder('/hd2/yangjiaxi/PDG_attack_imagenet/data/Imagenet/val',transform2)
data_loader = torch.utils.data.DataLoader(dataset,batch_size = Batchsize, shuffle = False,pin_memory = True)
data_loader2 = torch.utils.data.DataLoader(dataset,batch_size = Batchsize, shuffle = False,pin_memory = True)
attack_number = 0									#save number of attack
image_number = 0									#save number of images each attack
large_range = []
for zip1,zip2 in zip(enumerate(data_loader),enumerate(data_loader2)):
	j,data = zip1
	k,data2 = zip2
	image_number += 1
	if j%10 == 0:
		print("iteration:",j,"complete")
		print("correct number:",int(correct_sum[0]),"accuracy:",float(correct_sum[0]/j))
		print("failed attacked number:",correct_sum.view(255))
		print("out of range:",large_range)
	if j < 0:
		continue
	if j == 1:
		break
	cln_data, true_label = data
	cln_data2, true_label2 = data2

	img = transforms.ToPILImage()(cln_data.view(3,299,299)).convert('RGB')
	img.save('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result3/after_normalize{}.jpg'.format(image_number))
	img = transforms.ToPILImage()(unorm(cln_data).view(3,299,299)).convert('RGB')
	img.save('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result3/anti_normalize{}.jpg'.format(image_number))
	img = transforms.ToPILImage()(cln_data2.view(3,299,299)).convert('RGB')
	img.save('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result3/original{}.jpg'.format(image_number))
	eps = 1
	adversary = LinfPGDAttack(model, eps=float(eps)/255, eps_iter=float(eps)/255/40, nb_iter=40,rand_init=False, targeted=False)
	adv_untargeted = adversary.perturb(cln_data.cuda(), true_label.cuda())
	img = transforms.ToPILImage()(unorm(adv_untargeted.cpu()).view(3,299,299)).convert('RGB')
	img.save('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result3/after_attack{}.jpg'.format(image_number))


print(correct_sum)
print("out of range:",large_range)















