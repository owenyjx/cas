import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy
import scipy.misc

#load models
from torchvision.models import resnet101
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd

#load attacks
from advertorch.attacks import LinfPGDAttack

normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = resnet101(pretrained=True)
model.eval()
#model = nn.Sequential(normalize, model)
model = model.to(device)

#load data from validation of imagenet
Batchsize = 1
correct_sum = torch.zeros(17,1)
#transform = transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],std = [ 0.229, 0.224, 0.225 ]),])
transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),])
dataset = datasets.ImageFolder('/hd2/yangjiaxi/PDG_attack_imagenet/data/Imagenet/val',transform)
data_loader = torch.utils.data.DataLoader(dataset,batch_size = Batchsize, shuffle = False,pin_memory = True)
for j,data in enumerate(data_loader):
	if j%100 == 0:
		print("iteration:",j,"complete")
		print("correct number:",correct_sum[0],"accuracy:",float(correct_sum[0]/j))
		print("failed attacked number:",correct_sum.view(1,17))
	if j > 0:
		break
	cln_data, true_label = data
	cln_data, true_label = cln_data.to(device), true_label.to(device)
	pred_cln = predict_from_logits(model(cln_data)).to(device)
	print(cln_data)
	if pred_cln.cuda() == true_label:
		correct_sum[0] += 1
		for eps in range(1,17):
			 adversary = LinfPGDAttack(model, eps=float(eps)/255, eps_iter=float(eps)/255/40, nb_iter=40,rand_init=False, targeted=False)
			 adv_untargeted = adversary.perturb(cln_data.cuda(), true_label.cuda())
			 pred_untargeted_adv = predict_from_logits(model(adv_untargeted))
			 if pred_untargeted_adv == true_label:
			 	 correct_sum[eps] += 1
			 else:
			     break		 
print(correct_sum)
