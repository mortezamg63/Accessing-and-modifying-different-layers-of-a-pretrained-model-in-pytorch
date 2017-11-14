# Accessing and modifying different layers of a pretrained model in pytorch

The goal is dealing with layers of a pretrained Model like resnet18 to print and frozen the parameters. Let’s look at the content of resnet18 and shows the parameters. At first the layers separately are printed to see how we can access every layer individually. 

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

The following code print parameters for layers in a pretrained model.

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

Now, I consider to freeze the parameters of the first layer to the sixth layer as follow:

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
   
   ```
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
  
  In order to get some layers and remove the others, we can convert model.children() to a list and use indexing for specify which layers we want. For this purpose in pytorch, it can be done as follow:

   ```ruby
new_model = nn.Sequential(*list(model.children())[:-1])
   ```

The above line gets all layers except the last layer (it removes the last layer in model).

   ```ruby
new_model_2_removed = nn.Sequential(*list(model.children())[:-2])
   ```

The above line removes the two last layers in resnet18 and get others.



## Add new loss function 
To add a new loss function it is necessary to define a class that inherits from torch.nn.Module class. After  declaring initializing function, you just need to add forward function in order to compute loss and return it. In the following it is shown.

    
    class cust_loss(torch.nn.Module):
    def __init__(self):
        super(cust_loss, self).__init__()

    def forward(self, input, target):
        predicted_labels = torch.max(input, 1)[1].float()
        minus = predicted_labels - target.float()
        self.cust_distance = torch.sum(minus*minus).type(torch.FloatTensor)/predicted_labels.size()[0]
        return self.cust_distance

It is necessary that all lines in forward function to return FloatTensor type in order to autograd can be computed in pytorch backward function. Finally you must use the declared loss function in your main function during training as follow.

    ############ Withing main function ###########
    criterion = cust_loss()   #nn.CrossEntropyLoss()        
    Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=1e-3, momentum=0.9)
    ...
    loss = criterion(inputs, labels)
    loss.backward()
