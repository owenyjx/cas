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
from torchvision.models import resnet101
from torchvision.models import inception_v3
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd

#load attacks
from advertorch.attacks import LinfPGDAttack


model = resnet101(pretrained=True)
model.eval()
model = model.to(device)

model2 = inception_v3(pretrained=True)
model2.eval()
model2 = model.to(device)

#load data from validation of imagenet
Batchsize = 1
correct_sum = torch.zeros(255)
transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],std = [ 0.229, 0.224, 0.225 ]),])
transform2 = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),])
transform3 = transforms.Compose([transforms.Resize(299),transforms.CenterCrop(299),transforms.ToTensor(),transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],std = [ 0.229, 0.224, 0.225 ]),])

dataset = datasets.ImageFolder('/hd2/yangjiaxi/PDG_attack_imagenet/data/Imagenet/val',transform)
dataset2 = datasets.ImageFolder('/hd2/yangjiaxi/PDG_attack_imagenet/data/Imagenet/val',transform2)
dataset2 = datasets.ImageFolder('/hd2/yangjiaxi/PDG_attack_imagenet/data/Imagenet/val',transform3)

data_loader = torch.utils.data.DataLoader(dataset,batch_size = Batchsize, shuffle = False,pin_memory = True)
data_loader2 = torch.utils.data.DataLoader(dataset,batch_size = Batchsize, shuffle = False,pin_memory = True)
data_loader3 = torch.utils.data.DataLoader(dataset,batch_size = Batchsize, shuffle = False,pin_memory = True)

attack_number = 0									#save number of attack
image_number = 0									#save number of images each attack
tansition_attacked_number = 0
large_range = []

for zip1,zip2,zip3 in zip(enumerate(data_loader),enumerate(data_loader2),enumerate(data_loader3)):
	j,data = zip1
	k,data2 = zip2
	l,data3 = zip3

	image_number += 1
	if j%5 == 0:
		print("iteration:",j,"complete")
		print("correct number:",int(correct_sum[0]),"accuracy:",float(correct_sum[0]/j))
		print("tansition_attacked_number:",tansition_attacked_number)
		print("failed attacked number:",correct_sum.view(255))
		print("out of range:",large_range)
	if j < 30000:
		continue
	if j == 40000:
		break

	cln_data, true_label = data
	cln_data, true_label = cln_data.to(device), true_label.to(device)

	cln_data2, true_label2 = data2
	cln_data3, true_label3 = data3

	pred_cln = predict_from_logits(model(cln_data)).to(device)
	if pred_cln == true_label:
		correct_sum[0] += 1
		eps = 1
		while (True):
			adversary = LinfPGDAttack(model, eps=float(eps)/255, eps_iter=float(eps)/255/40, nb_iter=40,rand_init=False, targeted=False)
			adv_untargeted = adversary.perturb(cln_data.cuda(), true_label.cuda())
			pred_untargeted_adv = predict_from_logits(model(adv_untargeted))
			if pred_untargeted_adv == true_label:
				if eps < 255:
			 		correct_sum[eps] += 1
				else:
					large_range.append(eps)
			else:
				img = transforms.ToPILImage()(cln_data.cpu().view(3,224,224)).convert('RGB')
				img_attacked = transforms.ToPILImage()(unorm(adv_untargeted.cpu()).view(3,224,224)).convert('RGB')
				is_exist = os.path.exists('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result4/{}'.format(eps))
				if not is_exist:
					os.mkdir('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result4/{}'.format(eps))
				img.save('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result4/{}/{}.jpg'.format(eps,image_number))
				img_attacked.save('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result4/{}/{}_attacked.jpg'.format(eps,image_number))

				#transition
				adv_untargeted_transition = adversary.perturb(cln_data3.cuda(), true_label.cuda())
				pred_transition = predict_from_logits(model2(adv_untargeted_transition))
				if pred_transition != true_label:
					tansition_attacked_number += 1
				else:
					img.save('/hd2/yangjiaxi/PDG_attack_imagenet/gitupload/result4/{}/transition_attack_fail_{}.jpg'.format(eps,image_number))

				break	 
			eps += 1


print(correct_sum)
print("out of range:",large_range)















