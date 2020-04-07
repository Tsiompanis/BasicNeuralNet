#include<cstdio>
#include<vector>
#include<cstdlib>
#include<stdlib.h> 
#include<iostream>
#include<time.h>
#include<cmath>
//TODO:
//	Find a way to check if the number of input values is the same with the number of input neurons   RECENT AVG MEASURMENT (27.30)  
//CREATE TARGET VALS AND TRAINING DATA FUNCTIONS!!!

using namespace std;

vector<double> inputValues;
int networkSize; //the total number of layers in the whole network
int topology[50]; //how many Neurons each layer of the Neural Network has


//Reads the "topology.nn" file and writes the data in the topology array. For its structure read the file's description
void getTopology() 
{
	freopen("topology.nn","r",stdin);        
    scanf("%d",&networkSize);
    
    for(int i = 0; i < networkSize; i++) scanf(" %d", &topology[i]);
}


class Neuron; //this is just to be able to make a typedef early in the program (DONT REMOVE)

typedef vector<Neuron> Layer; //a layer is just an array of Neurons!
vector<Layer> allLayers; //Initializes a 2D vector of vectors of Neurons. (typedef vector<Neuron> Layer)

//###################################################################### class Neuron ######################################################################
struct connection //every Neuron connects with another (except the output neurons) and the weight and delta weight is needed.
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	//the initialization of a Neuron. It takes and int that states how many other Neurons it connects to and fills a dynamic array with connection structs
	Neuron(int numberOfOutputs, int currentIndex) 
	{
		for(int i = 0; i < numberOfOutputs; i++)
		{
			outputWeights.push_back(connection());
			outputWeights.back().weight = randomWeightGenerator();
		}
		neuronsIndex = currentIndex;
	}
	
	
	void setOutput(double val){currentOutput = val;} //sets its own currentValue to whatever is specified while calling the function
	
	double getOutputValue(void){ return currentOutput; } //(the Neuron) gives the caller its own output value
	
	void feedForward(Layer &prevLayer)
	{
		double sum = 0.0;
		
		//loops through all of the neurons in the previous layer and gets their output values and sums them.
		for(int n = 0; n < prevLayer.size(); n++)
		{
			sum = sum + prevLayer[n].getOutputValue() * prevLayer[n].outputWeights[neuronsIndex].weight;      //takes the output value * output weight and puts it in a sum value (look connection struct)
		}
		
		
		currentOutput = Neuron::activationFunction(sum); //Aply the magic activation function
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
			Neuron %neuron = prevLayer[n];
			
			double oldDeltaWeight = neuron.outputWeights[neuronsIndex].deltaWeight;
			
			double newDeltaWeight = 
		}
	}
	
	
private:
	
	double currentOutput; //every neuron has an output
	double currentGradient;
	vector<connection> outputWeights; //every neurons output has a weight and a delta weight (see struct connection)
	int neuronsIndex;
	
	double randomWeightGenerator()//returns a random double number between 0 and 1
	{ 
		double r = (double) rand()/RAND_MAX;
		printf("%lf ",r);
		return r;
	} 
	
	//the magic activation function tanh is used -> output range is from -1.0 to 1.0    <<---------
	double activationFunction(double x)
	{
		return tanh(x);
	}
	
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



//#################################################################### class Net ####################################################################
//typedef vector<Neuron> Layer; //a layer is just an array of Neurons!
//vector<Layer> allLayers; //Initializes a 2D vector of vectors of Neurons. (typedef vector<Neuron> Layer)
class Network
{
public:
	
	//Intialize the NeuralNet by filling a 2D vector with Neurons. (Each layer with the quantity of NEURONS specified in the topology)
	void initialize()
	{
		//From layer 0 up to layer "networkSize" (the total number of layers in the network).
		for(int layer = 0; layer < networkSize; layer++)
		{
			allLayers.push_back(Layer());
			
			
			//the numberOfOutputs integer states how many other Neurons a certain Neuron connects to.
			//the neurons at the output layer dont connect to other Neurons so we are intrested in all the other layers
			int numberOfOutputs = 0;
			if(layer != networkSize - 1) numberOfOutputs = topology[layer + 1];
		
			
			//For every layer's max number of Neurons (specified in the topology)
			for(int currentNeuron = 0; currentNeuron <= topology[layer]; currentNeuron++)
			{
				allLayers.back().push_back(Neuron(numberOfOutputs, currentNeuron)); //Pushing as many Neurons in the current layer as specified in the topology
				printf("New Neuron created!\n\n\n");				
			}
		}
	}
	
	//for every Neuron make the calculations
	void feedForward()
	{
		for(int i = 0; i < inputValues.size(); i++) //for every input value that we have set the coresponding node to the input value
		{
			allLayers[0][i].setOutput(inputValues[i]);
		}
		
		
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
	
	//
	void backPropagation()//(many maths :')
	{
		//Calculate overall net error (RMS of output neuron errors)
		Layer &outputLayer = allLayers.back();
		netError = 0.0;
		
		for(int n = 0; n < outputLayer.size(); n++)
		{
			double delta = targetVals[n] - outputLayer[n].getOutputValue();
			netError += delta * delta;
		}
		netError /= outputLayer.size() - 1;
		netError = sqrt(netError); //RMS
		
		//Calculate output layer gradients
		for(int n = 0; n < outputLayer.size(); n++)
		{
			outputLayer[n].calcOutputGradients(targetVals[n])
		}
		
	
		//Calculate gradients on hidden layers
		for(int layerNumber = allLayers.size() - 2; layerNumber > 0; layerNumber--)// for all hidden layers 
		{
			Layer &hiddenLayer = allLayers[layerNumber];
			Layer &nextLayer = allLayers[layerNum + 1];
			
			for(int n = 0; n < hiddenLayer.size(); n++)//for every neuron of every hidden layer.
			{
				hiddenLayer[n].calcHiddenGradients(nextLayer)
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
	
	
	
private:
	double netError;
};



int main()
{
	srand (time(NULL));

	getTopology(); //reads the "topology.nn" file and stores the data
	Network myNet;
	
	myNet.initialize(); //initializing a new NeuralNetwork as stated in the Network class

		
	return 0;
}
