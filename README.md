# Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch

The goal is dealing with layers of a pretrained Model like resnet18 to print and frozen the parameters. Let’s look at the content of resnet18 and shows the parameters. At first the layers separately are printed to see how we can access every layer individually. 
<p style="color:gray;">
child_counter = 0<br>
for child in model.children():<br>
&nbsp&nbsp&nbsp print(" child", child_counter, "is:")<br>
&nbsp&nbsp&nbsp print(child)<br>
&nbsp&nbsp&nbsp child_counter += 1<br>

 </p>
