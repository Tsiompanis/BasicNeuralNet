#include<cstdio>
#include<vector>
#include<cstdlib>
#include<stdlib.h> 
#include<iostream>
#include<time.h>
#include <Windows.h>
#include<cmath>
//TODO:
//RECENT AVG MEASURMENT (27.30)  
//Take the result values from the neurons and put them in an array


//main.cpp: In member function ‘void Neuron::updateInputWeights(Layer&)’:
//main.cpp:127:11: error: expected unqualified-id before ‘%’ token
//    Neuron %neuron = prevLayer[n];
//           ^
//main.cpp:129:28: error: ‘neuron’ was not declared in this scope
//    double oldDeltaWeight = neuron.outputWeights[neuronsIndex].deltaWeight;
//                            ^~~~~~

using namespace std;


double eta = 0.15; //overall training rate -> 0.0 slow, 0.2 medium, 1.0 crazy fast (and reckless)    [0.0->1.0]
double alpha = 0.5; //momentum (fraction of the previous delta weight) -> 0.0 No momentum, 0.5 moderate momentum   [0.0->1.0]
int networkSize; //the total number of layers in the whole network
int dataInLength; //how many lines of inout are in the learningdata.nn file
int dataInSize; //how many input variables in each data line
int dataOutSize; //how many output variables in each data line
int topology[500]; //how many Neurons each layer of the Neural Network has
double inputLearningData[100000005]; //input learning data array
double outputTargetData[100000005]; //target value data array
class Neuron; //this is just to be able to make a typedef early in the program (DONT REMOVE)
typedef vector<Neuron> Layer; //a layer is just an array of Neurons!
vector<Layer> allLayers; //Initializes a 2D vector of vectors of Neurons. (typedef vector<Neuron> Layer)
char ui; //a character to help with using the program
bool ui_bool; //a boolean to help with using the program

//Reads the "topology.nn" file and the "lerarningdata.nn" file and writes the data in the topology array. For its structure read the file's description. 
//and also reads some details about the learing data: how many lines of data its been given and how many input and output variables each line of data has
//this also helps for some basic error handling, if the input Neurons are not equal to the input variables or the output Neurons equal to the output variables then we have a big problem!
int getNetLearningDetails() 
{
	//reading details from the "topology.nn" file
	freopen("topology.nn","r",stdin);   
    scanf("%d",&networkSize);
    printf("Topology:\n   Number of layers: %d\n\t\t       ",networkSize); //printing the info about the network
    
    for(int i = 0; i < networkSize; i++) 
	{
		scanf(" %d", &topology[i]); //puts how many Neurons are in each layer in the array topology[]
		printf("%d + 1 bias,  ",topology[i]);
	}
    
    printf("\n\n");
    
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
	//return 0;	
	
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
	Neuron(int numberOfOutputs, int currentIndex, bool isBias) 
	{
		//for every Neuron in the next layer:
		for(int i = 0; i < numberOfOutputs; i++)
		{
			outputWeights.push_back(connection()); //makes a connection (structure) inside the outputWeights vector
			outputWeights.back().weight = randomWeightGenerator(); //it puts a random initial weight in each connection (structure)
		}
		neuronsIndex = currentIndex; //sets its own index
		if(isBias) currentOutput = 1.0;
	}
	
	void setOutput(double val){currentOutput = val;} //sets its own currentValue to whatever is specified while calling the function
	
	double getOutputValue(void){ return currentOutput; } //(the Neuron) gives the caller its own output value

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
	
	void calcOutputGradients(double targetValue)
	{
		double delta = targetValue - currentOutput;
		currentGradient = delta * Neuron::activationFunctionDerivative(currentOutput);
	}
	
	void calcHiddenGradients(Layer &nextLayer)
	{
		double dow = sumDOW(nextLayer);
		currentGradient = dow * Neuron::activationFunctionDerivative(currentOutput);
	}
	
	void updateInputWeights(Layer &prevLayer)
	{
		//modify the weights in the connection container
		
		for(int n = 0; n < prevLayer.size(); n++)
		{
			Neuron &neuron = prevLayer[n];
			
			double oldDeltaWeight = neuron.outputWeights[neuronsIndex].deltaWeight;
			
			double newDeltaWeight = 
								
								eta //overall training rate -> 0.0 slow, 0.2 medium, 1.0 crazy fast (and reckless)
								* neuron.getOutputValue()
								* currentGradient
								+ alpha // Add the momentum (fraction of the previous delta weight) -> 0.0 No momentum, 0.5 moderate momentum
								* oldDeltaWeight;
								
			neuron.outputWeights[neuronsIndex].deltaWeight = newDeltaWeight;
			neuron.outputWeights[neuronsIndex].weight += newDeltaWeight;
		}
	}
	
private:
	
	double currentOutput; //every neuron has an output value
	double currentGradient;
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
	double activationFunctionDerivative(double x)
	{
		return 1.0 - x * x;
	}
	
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
	Network()
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
				allLayers.back().push_back(Neuron(numberOfOutputs, currentNeuron, false)); //Pushing as many Neurons in the current layer as specified in the topology	
			}
			
				allLayers.back().push_back(Neuron(numberOfOutputs, currentNeuron+1, true)); //Forces the value of the bias Neuron to 1
		}
		printf("Neurons Created...\n");
	}
	
	
	
		//for every Neuron make the calculations
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
		
//		printf("\nNeuralNet prediction:\n  ");
//		for(int i = 0; i < dataOutSize; i++)
//		{
//			printf("%lf",allLayers[networkSize][i].getOutputValue());
//		}
	}
	
	
		void backPropagation()//(many maths :')
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
	
	void testCustom()
	{
		printf("testing");
	}
	
private:
	double netError;
};
//############################################################################## class Net End ##############################################################################


//reads one line of input learning data and the corresponding target output values
void fetchInputLearningDataTargetData()
{
	//freopen("learningdata.nn","r",stdin); //read from the "learningdata.nn" file
	
	printf("\nTraining input(s):\n  ");
	for(int i = 0; i < dataInSize; i++)
	{
		double val;
		scanf("%lf",&val); //Read one line of the input learning data
		printf("%lf\n  ",val);
		allLayers[0][i].setOutput(val); //set the output value of the Neuron in layer 0 in the i-th position
	}
	
	printf("\nTarget output(s):\n  ");
	for(int i = 0; i < dataOutSize; i++)
	{
		scanf("%lf",&outputTargetData[i]); //Read one line of the target learing data
		printf("%lf\n  ",outputTargetData[i]);
	}
}



int main()
{
	srand (time(NULL));
	
	int error;
	error = getNetLearningDetails(); //gets some data needed for the learing such as how many layers will the Net have and how many Neurons will Exist in each layer
	if(error) return 0; //if an error is detected exit the program 


	
	Network myNet; //initialize a new Network
	
	
	for(int i = 0; i < dataInLength; i++){  //dataInLength
		fetchInputLearningDataTargetData();

		myNet.feedForward();
	
		printf("\nPrediction:\n	 %lf\n",allLayers[2][0].getOutputValue());
	
		myNet.backPropagation();
	}
	
	
//	printf("\n\n\nDo you wish to test? (Y/N):\n");
//	scanf("%c",&testing);
//	if(testing == 'y' || testing == 'Y') myNet.testCustom();
	
	
	return 0;
}
