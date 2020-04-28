// https://github.com/Tsiompanis/BasicNeuralNet
#include<cstdio>
#include<vector>
#include<cstdlib>
#include<stdlib.h> 
#include<iostream>
#include<time.h>
#include <Windows.h>
#include<cmath>

using namespace std;

double eta = 0.15; //overall training rate -> 0.0 slow, 0.2 medium, 1.0 crazy fast (and reckless)    [0.0->1.0]   0.15 is good
double alpha = 0.5; //momentum (fraction of the previous delta weight) -> 0.0 No momentum, 0.5 moderate momentum   [0.0->1.0]   0.5 is good
int networkSize; //the total number of layers in the whole network
int dataInLength; //how many lines of inout are in the learningdata.nn file
int dataInSize; //how many input variables in each data line
int dataOutSize; //how many output variables in each data line
int topology[500]; //how many Neurons each layer of the Neural Network has
double outputTargetData[100000005]; //target value data array
double errors[100000005]; //an array to hold the error of each time we feed forward
class Neuron; //this is just to be able to make a typedef early in the program (DONT REMOVE)
typedef vector<Neuron> Layer; //a layer is just an array of Neurons!
vector<Layer> allLayers; //Initializes a 2D vector of vectors of Neurons. (typedef vector<Neuron> Layer)
char import; //a character for importing an already existing model (Y/N input)
char printing_char; //a character to determine if the user wants to print while training (Y/N input)
bool printing; //a boolean to determine if the user wants to print while training
bool importFromFile; //a boolean to determine if the user wants to import an already trained network instad of making a new one

//Reads the "topology.nn" file and the "lerarningdata.nn" file and writes the data in the topology array. For its structure read the file's description. 
//and also reads some details about the learing data: how many lines of data its been given and how many input and output variables each line of data has
//this also helps for some basic error handling, if the input Neurons are not equal to the input variables or the output Neurons equal to the output variables then we have a big problem!
int getNetLearningDetails() 
{
	//reading details from the "topology.nn" file
	freopen("topology.nn","r",stdin);   
    scanf("%d",&networkSize);
    printf("\nThe structure of the Neural Net:\n   Number of layers: %d\n\n\n",networkSize); //printing the info about the network
    
    for(int i = 0; i < networkSize; i++) 
	{
		printf("\t\tLayer #%d\n",i+1);
		scanf(" %d", &topology[i]); //puts how many Neurons are in each layer in the array topology[]
		printf("\t\t%d neurons + 1 bias,\n\n",topology[i]);
	}
    
    
    //reading details from the learningdata.nn file 
    freopen("learningdata.nn","r",stdin);
	scanf("%d %d %d",&dataInLength,&dataInSize,&dataOutSize);
	
	//basic error handling
	if(topology[0] != dataInSize){
		printf("\nThe number of input Neurons is not equal to the number of input data!\nexiting the program...");
		return 1;
	}
	else if(topology[networkSize - 1] != dataOutSize){
		printf("\nThe number of output Neurons is not equal to the number of output data!\nexiting the program...");
		return 1;
	}
	return 0;	
	
}


//############################################################################## class Neuron ##############################################################################
struct connection //every Neuron connects with another (except the output neurons) and the weight and delta weight is needed.
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	//the birth (initialization) of a single Neuron. It takes and integer that states how many other Neurons it connects to and fills a dynamic array with connection structs
	Neuron(int numberOfOutputs, int currentIndex, bool isBias, bool import) 
	{
		//for every Neuron in the next layer:
		for(int i = 0; i < numberOfOutputs; i++)
		{
			outputWeights.push_back(connection()); //makes a connection (structure) inside the outputWeights vector
			
			if(!import) outputWeights.back().weight = randomWeightGenerator(); //if the neuron beeing created is not from an (import) file it puts a random initial weight in each connection (structure).
			
			else //if the neuron beeing created is imported from a file then 
			{
				double h;
				scanf("%lf",&h); //read a weight value from the import file ("model.nn")
				outputWeights.back().weight = h; //and set the current neurons weight to the read value
			}
		}
		neuronsIndex = currentIndex; //sets its own index
		if(isBias) currentOutput = 1.0; // if the neuron that is currently being created is a bias neuron it gets a forced output of 1.0
	}
	
	void setOutput(double val){currentOutput = val; } //sets its own currentValue to whatever is specified while calling the function
	
	double getOutputValue(void){ return currentOutput; } //(the Neuron) gives the caller its own output value
	
	double getWeights(int whatNeuron){ return outputWeights[whatNeuron].weight; } //returns the weight specified
	
	// sums the previous neurons multiplied by the weights and aplies an activation function (tanh) and sets its own output to the calculated sum
 	void feedForward(Layer &prevLayer)
	{
		double sum = 0.0;
		//loops through all of the neurons in the previous layer and gets their output values and sums them.
		for(int n = 0; n < prevLayer.size(); n++)
		{
			sum = sum + prevLayer[n].getOutputValue() * prevLayer[n].outputWeights[neuronsIndex].weight;   //takes the output value * output weight and puts it in a sum value (look connection struct)
		}
		
		currentOutput = Neuron::activationFunction(sum); //Aply the activation function (tanh)
	}
	
	//calculates the output layer's gradient
	void calcOutputGradients(double targetValue)
	{
		double delta = targetValue - currentOutput;
		currentGradient = delta * Neuron::activationFunctionDerivative(currentOutput);
	}
	
	//calculates the hidden layer's gradient
	void calcHiddenGradients(Layer &nextLayer)
	{
		double dow = sumDOW(nextLayer);
		currentGradient = dow * Neuron::activationFunctionDerivative(currentOutput);
	}
	
	//a function to updtate a neurons input weights during backpropagation (how the network fundementaly changes)
	void updateInputWeights(Layer &prevLayer)
	{
		//for every previous layers neuron
		for(int n = 0; n < prevLayer.size(); n++)
		{
			Neuron &neuron = prevLayer[n];
			
			double oldDeltaWeight = neuron.outputWeights[neuronsIndex].deltaWeight; //sets the old delta weight to be the current delta weight
			
			double newDeltaWeight = 
								
								eta //overall training rate -> 0.0 slow, 0.2 medium, 1.0 crazy fast (fraction of the previous neuron's output value)
								* neuron.getOutputValue()
								
								* currentGradient // multiply by the gradient
								
								+ alpha // Add the momentum (fraction of the previous delta weight) -> 0.0 No momentum, 0.5 moderate momentum
								* oldDeltaWeight;
								
			neuron.outputWeights[neuronsIndex].deltaWeight = newDeltaWeight; 
			neuron.outputWeights[neuronsIndex].weight += newDeltaWeight; //sets one of the output weights of one of the previous layer's neuron's weight (the one that connects with the current neuron)
		}
	}
	
private:
	
	double currentOutput; //every neuron has an output value
	double currentGradient; //the gradient of the neuron
	vector<connection> outputWeights; //every neurons output has a weight and a delta weight (see struct connection)
	int neuronsIndex; //the index of the Neuron inside a layer
	
	//returns a random double number between 0 and 1
	double randomWeightGenerator()
	{ 
		double r = (double) rand()/RAND_MAX;
		return r;
	} 
	
	double activationFunction(double x){ return tanh(x);} //the activation function tanh is used -> output range is from -1.0 to 1.0 
	
	//tanh derivative
	double activationFunctionDerivative(double x){ return 1.0 - x * x; } 
	
	
	double sumDOW(Layer &nextLayer)
	{
		double sum = 0.0;
		
		//sum our contributions of the errors at the nodes we feed
		for(int n = 0; n < nextLayer.size() - 1; n++)
		{
			sum += outputWeights[n].weight * nextLayer[n].currentGradient;
		}
		
		return sum;
	}
};
//############################################################################## class Neuron END ##############################################################################




//################################################################################ class Net ###################################################################################
class Network
{
public:
	
	//Intialize the NeuralNet by filling a 2D vector with Neurons. (Each layer with the quantity of NEURONS specified in the topology)
	void initialize(bool import)
	{
		//From layer 0 up to layer "networkSize" (the total number of layers in the network).
		for(int layer = 0; layer < networkSize; layer++)
		{
			allLayers.push_back(Layer());
			int currentNeuron;
			
			//the numberOfOutputs integer states how many other Neurons a certain Neuron connects to.
			//the neurons at the output layer dont connect to other Neurons so we are intrested in all the other layers
			int numberOfOutputs = 0;
			if(layer != networkSize - 1) numberOfOutputs = topology[layer + 1];
		
			
			//For every layer's max number of Neurons (specified in the topology)
			for(currentNeuron = 0; currentNeuron < topology[layer]; currentNeuron++)
			{
				allLayers.back().push_back(Neuron(numberOfOutputs, currentNeuron, false, import)); //Pushing as many Neurons in the current layer as specified in the topology	
			}
			
				allLayers.back().push_back(Neuron(numberOfOutputs, currentNeuron+1, true, import)); //Forces the value of the bias Neuron to 1
		}
		printf("\n\nYour network was created successfully...\n");
	}
	
	
	
	//for every Neuron in the network feed forward) 
	void feedForward()
	{
		//Move forward (for every Layer, for every Neuron of each layer) (starting from the second layer because the input layer does not make any calculations)
		for(int layerNumber = 1; layerNumber < networkSize; layerNumber++)
		{
			Layer &prevLayer = allLayers[layerNumber - 1];
			for(int neuronNumber = 0; neuronNumber < allLayers[layerNumber].size() - 1; neuronNumber++) //for every layer's Neuron except the bias Neuron
			{
				allLayers[layerNumber][neuronNumber].feedForward(prevLayer);
			}
		}
		

	}
	
	
	//go back to change all the weights so the output is more optimal
	void backPropagation()
	{
		//Calculate overall net error (RMS of output neuron errors)
		Layer &outputLayer = allLayers.back();
		netError = 0.0;
		
		for(int n = 0; n < outputLayer.size()-1; n++)
		{
			double delta = outputTargetData[n] - outputLayer[n].getOutputValue();
			netError += delta * delta;
		}
		netError /= outputLayer.size() - 1;
		outputNetError = netError;
		netError = sqrt(netError); //RMS
		
		//Calculate output layer gradients
		for(int n = 0; n < outputLayer.size(); n++)
		{
			outputLayer[n].calcOutputGradients(outputTargetData[n]);
		}
		
	
		//Calculate gradients on hidden layers
		for(int layerNumber = allLayers.size() - 2; layerNumber > 0; layerNumber--)// for all hidden layers 
		{
			Layer &hiddenLayer = allLayers[layerNumber];
			Layer &nextLayer = allLayers[layerNumber + 1];
			
			for(int n = 0; n < hiddenLayer.size(); n++)//for every neuron of every hidden layer.
			{
				hiddenLayer[n].calcHiddenGradients(nextLayer);
			}
			
		}
		
	
		//For all layers starting form outputs to first hidden layer, update the connection weights
		for(int layerNumber = allLayers.size() - 1; layerNumber > 0; layerNumber--)
		{
			Layer &layer = allLayers[layerNumber];
			Layer &prevLayer = allLayers[layerNumber - 1];
			
			for(int n = 0; n < layer.size() - 1; n++)
			{
				layer[n].updateInputWeights(prevLayer);
			}
		}
	}
	
	//prints the value of the output neurons
	void printOutput()
	{
		freopen("CON","w",stdout);
		printf("\nNeuralNet prediction:\n ");
		//for every output neuron print its output value
		for(int j = 0; j < topology[networkSize-1]; j++)
		{
			double out = allLayers[networkSize-1][j].getOutputValue();
			printf(" %lf ",out);
			cout << "(rounded: " << round(out) << ")" << "\n";

		}
		printf("**************************");
	}
	
	// function to test the network with custom data
	void testCustom()
	{
		freopen("CON","r",stdin);  //read from the console
		freopen("CON","w",stdout); //write in the console
		printf("\n\nTesting...\n");
		for(int i = 0; i < allLayers[0].size() - 1; i++) //for every input Neuron
		{
			double val;
			scanf("%lf",&val); //Read one line of the input learning data
			allLayers[0][i].setOutput(val); //set the output value of the Neuron in layer 0 in the i-th position
		}
		
		feedForward(); //determine the output
		
		printOutput(); //print the output
		 
		freopen("CON","r",stdin); //read from the console
		//determine if the user wants to test the network again
		printf("\n\n\nDo you want to test again? (Y/N): ");
		char testing;
		scanf("%c",&testing);
		if(testing == 'y' || testing == 'Y') testCustom();
	}
	
	void exportModel()
	{
		freopen("topology_model.nn","w",stdout); //write in the "topology_model.nn" file
		printf("%d\n", networkSize); //write how many layers the network being exported has
		for(int i = 0; i < networkSize; i++) printf("%d ",topology[i]); //for every layer write how many Neurons it has
		
		freopen("model.nn","w",stdout); //write in the "exported_mode.nn" file
		for(int layer = 0; layer < networkSize; layer++) //for every layer of the network being exported
		{
			for(int n = 0; n <= topology[layer]; n++) //for every Neuron of the layer of the network that is being exported 
			{
				for(int i = 0; i < topology[layer+1]; i++) // for every connection this layer has
				{
					double w = allLayers[layer][n].getWeights(i); //write the weight in the export file
					printf("%lf ",w);
				}
			}
			printf("\n"); //write an empty line for visual clarity
		}
		
		
		freopen("CON","w",stdout); //write in the console
		printf("\nModel exported in the file topology_model.nn and the file model.nn..."); 
	}
	
	double getNetError(){ return outputNetError; } //returns the net error
	
private:
	double outputNetError;
	double netError; //the overall network error
	
};
//############################################################################## class Net End ##############################################################################


//reads one line of input learning data and the corresponding target output values
void fetchInputLearningDataTargetData(bool printing) // forgive me for this name
{
	if(printing) printf("\nTraining input(s):\n  "); //if the user has enabled printing, print some info 
	for(int i = 0; i < dataInSize; i++)
	{
		double val;
		scanf("%lf",&val); //Read one line of the input learning data
		if(printing) printf("%lf\n  ",val); //if the user has enabled printing, print some info 
		allLayers[0][i].setOutput(val); //set the output value of the Neuron in the input layer in the i-th position to an input read from the training input file
	}
	
	if(printing) printf("\nTarget output(s):\n  "); //if the user has enabled printing, print some info 
	for(int i = 0; i < dataOutSize; i++)
	{
		scanf("%lf",&outputTargetData[i]); //Read one line of the target learing data
		if(printing) printf("%lf\n  ",outputTargetData[i]); //if the user has enabled printing, print some info 
	}
}


void getTrainedModelTopology()
{
	freopen("topology_model.nn","r",stdin);
	scanf("%d",&networkSize);
    printf("\nTopology:\n   Number of layers: %d\n\t\t       ",networkSize); //printing the info about the network
    
    for(int i = 0; i < networkSize; i++) 
	{
		scanf(" %d", &topology[i]); //puts how many Neurons are in each layer in the array topology[]
		printf("%d + 1 bias,  ",topology[i]); //prints info about the topology of the neural network
	}
}

//******************************************************************************* main *******************************************************************************
int main()
{
	srand (time(NULL));
	
	// find out if the user wants to import a model instead of making a new one
	printf("Do you want to import an already existing model? (NOTE: the model will be imported from the file 'model.nn')(Y/N): ");
	scanf("%c",&import);
	if(import == 'y' || import == 'Y')
	{
		importFromFile = true;
		getTrainedModelTopology(); //gets the topology from the "topology_model.nn" file
		freopen("model.nn","r",stdin); //from now on it reads from the "model.nn" file
	}
	
	
	if(!importFromFile) //if the user wants to create a new Neural Net do:
	{
		int error = getNetLearningDetails(); //gets some data needed for the learing such as how many layers will the Net have and how many Neurons will Exist in each layer
		if(error) return 0; //if an error is detected exit the program 
	}
	
	Network myNet; //initialize a new Network
	myNet.initialize(importFromFile);
	
	//find out if the user wants to print the data while training the neural network (makes the training process much slower)
	if(!importFromFile) //if the user wants to create a new Neural Net do:
	{
		freopen("CON","r",stdin); // read from the console
		printf("\n\nDo you want to print information about the learning proccess while training the current model(makes training much slower!)? (Y/N): ");
		scanf("%c",&printing_char);
		if(printing_char == 'y' || printing_char == 'Y') printing = true; //if yes then enable printing while training
	}
	
	
	if(!importFromFile)
	{
		freopen("CON","r",stdin); //read from the console to determine if the user wants to train with a custom training rate
		freopen("CON","w",stdout);
		printf("\nDo you want to train the model with a custom training rate and delta? (Y/N): ");
		char ct;
		scanf("%c",&ct);
		if(ct == 'Y' || ct == 'y')
		{
			printf("Overall training rate (eta) [0.0->1.0] (0.15 is good): ");
			scanf("%lf",&eta); //reading custom eta value from console
			printf("\nMomentum (fraction of the previous delta weight) [0.0->1.0] (0.5 is good):");
			scanf("%lf",&alpha); //reading custom alpha value from console
		}
		
		freopen("learningdata.nn","r",stdin); //read from the "learningdata.nn" file
		int h;
		scanf("%d %d %d",&h,&h,&h); //this line is needed for the fetchInputLearningDataTargetData to work properly (the worst patch in human history)
		
		for(int i = 0; i < dataInLength; i++){ // for every line of input data we have
		
			if(printing) printf("\n\nPass #%d",i+1); //if the user has enabled printing, print some info 
		
			fetchInputLearningDataTargetData(printing); //read the training data (the input and the target)

			myNet.feedForward(); //feed forward the training data
			errors[i] = myNet.getNetError();
		
			if(printing) myNet.printOutput(); //if the user has enabled printing, print what the network predicts
		
			myNet.backPropagation(); //back propagate the network to improve it
		}
	}
	
	freopen("model-accuracy.nn","w",stdout); //Print the accuracy of the Neural Net in a file 
	for(int i = 0; i < dataInLength; i += 10) 
	{
		double sum;
		for(int j = 0; j < 10; j++) sum += errors[i+j];
		printf("%lf\n",sum/10);
	}
	
	freopen("CON","r",stdin); //read from the console to determine if the user wants to test the trained network
	freopen("CON","w",stdout);
	printf("\n\n\nDo you want to test the trained model? (Y/N): ");
	char testing;
	scanf("%c",&testing);
	if(testing == 'y' || testing == 'Y') myNet.testCustom(); //if yes call the testCustom function
	
	
	freopen("CON","r",stdin); //read from the console to determine if the user wants to export the neural network!
	freopen("CON","w",stdout);
	printf("\n\n\nDo you want to export this model into a file? (Y/N): ");
	char exporting;
	scanf("%c",&testing);
	if(testing == 'y' || testing == 'Y') myNet.exportModel(); //if yes call the export model function
	
	return 0; //exit the program
}
