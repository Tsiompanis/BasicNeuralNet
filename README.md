# Basic Neural Network implementation in C++

**About:** A program to create, train, test, and export artificial Neural Networks.


## How to use:

To create and train a new Neural Network you must have two files: **"topology.nn"** & **"learningdata.nn"**	You can do that simply by executing the **"createfiles.exe"** program!



----------

#### How the "topology.nn" file works:

In the first line of the topology file you must state **how many layers deep** you want your Neural Network to be (N). The second line must contain **N integers (one for every layer) that state how many Neurons you want each layer to have**. **For Example:**

```4			The Neural Network is going to have 4 layers 
4	The Neural Network is going to have 4 layers 

4 5 3 2	The first layer is going to have 4 Neurons, the second layer 5 the third layer 3 and so on...	
```

----------



After the "topology.nn" file is created you must edit the **"learningdata.nn"** file.



---------

#### How the "learningdata.nn" file works:

In the first line of the learningdata you must state: How how large (N) your data set is (see example), how many inputs you give (I) and how many target outputs outputs you give (O).

The rest N lines contain inputs and outputs. **Example:**

```
4 2 1	The dataset has 4 lines of training data and each line has 2 inputs(I) and 1 target output(O)

1 1 0	The first line of training data has the inputs 1 & 1 and one target output 0
0 0 0	The second line of training data has the inputs 0 & 0 and one target output 0
1 0 1	The third line of training data has the inputs 1 & 0 and one target output 1
0 1 1	The fourth line of training data has the inputs 0 & 1 and one target output 1
```

**NOTE: THE INPUT-OUTPUT VALUES MUST BE BETWEEN 0 AND 1**

**NOTE: THE INPUT-OUTPUT VALUES MUST BE BETWEEN 0 AND 1**

-----



**After the files are complete you can execute the "BasicNeuralNet.V.7.exe" file where the model can be trained, tested and exported (and imported)!**



**NOTE: when you export files make sure you don't have export files in the folder you run the program because these files WILL BE OVERWRITTEN.**

  

###### April 2020  [Peter Tsiompanis](https://tsiompanis.com/)
