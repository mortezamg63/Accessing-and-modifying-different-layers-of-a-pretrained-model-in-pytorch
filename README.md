# Notes in pytorch to deal with ConvNets


## Accessing and modifying different layers of a pretrained model in pytorch

The goal is dealing with layers of a pretrained Model like resnet18 to print and frozen the parameters. Let’s look at the content of resnet18 and shows the parameters. At first the layers are printed separately to see how we can access every layer individually. 

```ruby

child_counter = 0
for child in model.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1
```

The output of this piece of code is as follow:

```
child 0 is -
Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
child 1 is:
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
child 2 is:
ReLU (inplace)
child 3 is:
MaxPool2d (size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1))
child 4 is:
Sequential (
(0): BasicBlock ((conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
(relu): ReLU (inplace)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
)
(1): BasicBlock ((conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
(relu): ReLU (inplace)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
)
)
child 5 is:
Sequential (
(0): BasicBlock ((conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1,1),bias=False)
(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
(relu): ReLU (inplace)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
(downsample): Sequential (
(0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
)
)
(1): BasicBlock (
(conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
(relu): ReLU (inplace)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
)
)
child 6 is: 
        others are not shown because of consuming less space
```

## Freezing parameters of some layers to prevent them from retraining

The following code prints parameters for layers in a pretrained model.

```ruby
for child in model.children():
   for param in child.parameters():
      print(param)
      break
   break
```

Several lines of output are shown as follow:

```
(0 ,0 ,.,.) =
1.8160e-02 2.1680e-02 5.6358e-02       
-02
2.6440e-02 1.0603e-02 1.9794e-02
-02
9.0205e-03 1.9536e-03 1.9925e-04
-03
...
-2.4830e-02 8.1022e-03 -4.9934e-02
-02
-2.3857e-02 -1.6275e-02 2.9058e-02
-03
-1.6848e-04 5.9266e-02 -5.8456e-03
-02
(0 ,1 ,.,.) =
-1.6319e-02 3.3193e-02 -2.2146e-04
```

Now, I consider to freeze the parameters of the first to the sixth layers as follow:

```ruby
child_counter = 0
for child in model.children():
   if child_counter < 6:
      print("child ",child_counter," was frozen")
      for param in child.parameters():
          param.requires_grad = False
   elif child_counter == 6:
      children_of_child_counter = 0
      for children_of_child in child.children():
         if children_of_child_counter < 1:
            for param in children_of_child.parameters():
	param.requires_grad = False
	print('child ', children_of_child_counter, 'of child',child
		_counter,' was frozen')
      else:
         print('child ', children_of_child_counter, 'of child',child
         _counter,' was not frozen')
         children_of_child_counter += 1
   else:
      print("child ",child_counter," was not frozen")
      child_counter += 1
   ```
The output of above code is as follow:   
   
   ```ruby
   child  0 was frozen
child  1 was frozen
child  2 was frozen
child  3 was frozen
child  4 was frozen
child  5 was frozen
child  0 of child 6 was frozen
child  1 of child 6 was not frozen
child  7 was not frozen
child  8 was not frozen
child  9 was not frozen
   ```
   
  ## Getting some layers
  
  In order to get some layers and remove the others, we can convert model.children() to a list and use indexing for specifying which layers we want. For this purpose in pytorch, it can be done as follow:

   ```ruby
     new_model = nn.Sequential(*list(model.children())[:-1])
   ```

The above line gets all layers except the last layer (it removes the last layer in model).

   ```ruby
   new_model_2_removed = nn.Sequential(*list(model.children())[:-2])
   ```

The above line removes the two last layers in resnet18 and get others.



## Adding new loss function 
To add a new loss function it is necessary to define a class that inherits from torch.nn.Module class. After  declaring initialization function, you just need to add forward function in order to compute loss and return it. In bottom it is shown.
```ruby
   class cust_loss(torch.nn.Module):
	    def __init__(self):
		super(cust_loss, self).__init__()

	    def forward(self, input, target):
		predicted_labels = torch.max(input, 1)[1].float()
		minus = predicted_labels - target.float()
		self.cust_distance = torch.sum(minus*minus).type(torch.FloatTensor)/predicted_labels.size()[0]
		return self.cust_distance
   ```
	
It is necessary note that all lines in forward function must return FloatTensor type in order to autograd can be computed in backward function. Finally you must use the declared loss function in your main function during training as follow.

   ```ruby
   ############ Withing main function ###########
    criterion = cust_loss()   #nn.CrossEntropyLoss()        
    Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=1e-3, momentum=0.9)
    ...
    loss = criterion(inputs, labels)
    loss.backward()
   ```

## Getting data from dataloader
It is considered to get data from dataloader without loop statement. There is a function that can do this for us. 

   ```ruby
   iter(train_target_loader).next()
   ```	

This statment returns a tensor with the size of 2\*batch_size*size_of_data. The first column is loaded data and the second column is corresponding labels for loaded data. Therefore, if we use the following statement, It returns data. It returns labels in case we change index to 1.

   ```ruby
   (iter(train_target_loader).next())[0]
   ```	 

## Manipulation a pretrained model

Sometimes it is needed to extract some features from different layers of a pretrained model in a way that forward function can be run one time. In other words, running forward function in pretrained model and stopping it in a layer whose output is our interest is not a good method. Assume you wants to get output of several layers and you must run forward function several times (ie the number of runs is the number of layers whose output is our interest). To achieve this goal it is needed some background information.

Now consider the VGG16 architecture that is as follow (it is output of python)

	VGG (
	  (features): Sequential (
	    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (1): ReLU (inplace)
	    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (3): ReLU (inplace)
	    (4): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (6): ReLU (inplace)
	    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (8): ReLU (inplace)
	    (9): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (11): ReLU (inplace)
	    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (13): ReLU (inplace)
	    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (15): ReLU (inplace)
	    (16): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (18): ReLU (inplace)
	    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (20): ReLU (inplace)
	    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (22): ReLU (inplace)
	    (23): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (25): ReLU (inplace)
	    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (27): ReLU (inplace)
	    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (29): ReLU (inplace)
	    (30): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
	  )
	  (classifier): Sequential (
	    (0): Linear (25088 -> 4096)
	    (1): ReLU (inplace)
	    (2): Dropout (p = 0.5)
	    (3): Linear (4096 -> 4096)
	    (4): ReLU (inplace)
	    (5): Dropout (p = 0.5)
	    (6): Linear (4096 -> 1000)
	  )
	)	

To show you how to do this task, I use an example for illustration. Assume that I want to extract the first layers of VGG16 as features. In this regard, look at the following picture. The blue line shows which outputs I consider to get from layers.

![vgg-short](https://user-images.githubusercontent.com/15813546/32988686-5119820c-cd1e-11e7-8213-7a21a3227863.png)

As it can be seen from above picture and python output, our desire part of  vgg net is lines in the python output correspond with line from (0) to (15). Also, we need to concatenate output of lines (3), (8) and (15). The outputs of lines (8) and (15) must be enlarged (upsample) to obtain the size of the output in line (3); then they are concatenated to acheive the result. 

Now implementing a class for this purpose is as follow:
   ```ruby
   class myModel(nn.Module):
	    def __init__(self):
		super(myModel,self).__init__()
		vgg_model = torchvision.models.vgg16(pretrained=True)
		for child in vgg_model.children():
		    self.Conv1 = child[0]  # 3->64
		    self.Conv2 = child[2]  # 64->64
		    self.Conv3 = child[5]  # 64->128
		    self.Conv4 = child[7]  # 128->128
		    self.Conv5 = child[10]  # 128->256
		    self.Conv6 = child[12]  # 256->256
		    self.Conv7 = child[14]  # 256->256
		    self.upSample1 = nn.Upsample(scale_factor=2)
		    self.upSample2 = nn.Upsample(scale_factor=4)
		    break
	    def forward(self,x):
		out1 = self.Conv1(x)
		out1 = F.relu(out1)
		out1 = self.Conv2(out1)
		out1 = F.relu(out1)
		out1_mp = F.max_pool2d(out1, 2, 2)
		out2 = self.Conv3(out1_mp)
		out2 = F.relu(out2)
		out2 = self.Conv4(out2)
		out2 = F.relu(out2)
		out2_mp = F.max_pool2d(out2, 2, 2)
		out3 = self.Conv5(out2_mp)
		out3 = F.relu(out3)
		out3 = self.Conv6(out3)
		out3 = F.relu(out3)
		out3 = self.Conv7(out3)
		out3 = F.relu(out3)
		###### up sampling to create output with the same size
		out2 = self.upSample1(out2)
		out3 = self.upSample2(out3)
		#out7_mp = F.max_pool2d(out7, 2, 2)
		concat_features = torch.cat([out1, out2, out3], 1)
		return out1, concat_features
   ```
   
Also the above calss can be defined as follow:

   ```ruby
   class myModel(nn.Module):
	    def __init__(self):
		super(myModel,self).__init__()
		vgg_model = torchvision.models.vgg16(pretrained=True)
		for child in vgg_model.children():
		    self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
               self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9]) 
               self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
		    self.upSample1 = nn.Upsample(scale_factor=2)
		    self.upSample2 = nn.Upsample(scale_factor=4)
		    break
	    def forward(self,x):
		out1 = self.Conv1(x)
      	out2 = self.Conv2(out1)
        	out3 = self.Conv3(out2)
		###### up sampling to create output with the same size
		out2 = self.upSample1(out2)
		out3 = self.upSample2(out3)
		concat_features = torch.cat([out1, out2, out3], 1)
		return out1, concat_features
   ```
		
I hope this piece of code can be helpful :-)

## Adding custom transformation to dataloader
When we want to do a kind of preprocessing that is not implemented in pytorch library, it is axiomatic that preprocessing must be done on loaded images similar to torchvision.Transforms. But there is no function to do what we want. In this regard, we must implement our function in order to do preprocessing in a way that pytorch is going. So, I want to show you implementing a custom trasformation. (Wow, it is an easy job :-). So easy that it can be unbelievable)

At first, consider that our transformation is as easy as changing all pixel' values to zero. Assume that all images' pixels must be zero without considering a random choice. We want to put the function that do this task in code to be done as others torch transformations. For this purpose, a class is declared as follow, pay attention to inheritance from object class.

 ```ruby
   class myCustomTransform(object):
    def __call__(self, img):        
        new_img_data = []
        for color in img.getdata():
            new_img_data.append((0, 0, 0))

        newimg = Image.new(img.mode, img.size)
        newimg.putdata(new_img_data)
        return newimg
   ```
   After the definition of the class, we can use in source code with other torch transformations and change images to black images as follow.

 ```ruby
   data_transforms = {
        'train': transforms.Compose([            
            transforms.CenterCrop(224),
            myCustomTransform(),
            transforms.ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
	
    train_data_dir = 'xxxx/'
    val_data_dir   = 'xxxx/'
    train_set = datasets.ImageFolder(os.path.join(train_data_dir, 'train'), data_transforms['train']) #train    
    val_set   = datasets.ImageFolder(os.path.join(val_data_dir, 'val'), data_transforms['val'])
   ```
In the above code transforms.Normalize is commented because when you debug your code, zero values in training images which are loaded as a batch will be unchanged. If the normalization is done, the zero values in image's pixels changes.
