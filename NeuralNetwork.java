package RecurrentNeuralNetwork;
import java.io.*;
import java.util.*;

import RecurrentNeuralNetwork.Node.Activation;
import RecurrentNeuralNetwork.Node.ActivationDer;

public class NeuralNetwork
{
   private int[] dim;
   private Node networkBack;
   private Node[] networkFront;
   private int derCounter;
   private double rate = 0.75;
   private List<double[]> weightData;
   private Activation activation;
   private ActivationDer activationDer;
   
   public NeuralNetwork(int[] dim)//creates networkBack with dimensions;
   {
	  derCounter = 0;
      networkFront = new Node[1];
      weightData = new ArrayList<double[]>();
      this.dim = dim;
      
      construct();
   }
   
   public NeuralNetwork(int[] dim, Activation activate, ActivationDer activateDer)//creates networkBack with dimensions, and with different activation;
   {
	  derCounter = 0;
      networkFront = new Node[1];
      weightData = new ArrayList<double[]>();
      this.dim = dim;
      activation = activate;
      activationDer = activateDer;
      
      construct();
   }
   
   public NeuralNetwork(int input, int output)
   {
	   derCounter = 0;
	   networkFront = new Node[1];
	   weightData = new ArrayList<double[]>();
	   dim = new int[(int) Math.log(Math.max(input, output)) + 3];
		
	   for(int i = 1; i < dim.length; i++)
		   dim[i] = input * (1 - i / (dim.length - 1)) + output * i / (dim.length - 1);
	   
	   construct();
   }
   
   public NeuralNetwork(int input, int output, Activation activate, ActivationDer activateDer)
   {
	   derCounter = 0;
	   networkFront = new Node[1];
	   weightData = new ArrayList<double[]>();
	   dim = new int[(int) Math.log(Math.max(input, output)) + 3];
	   activation = activate;
	   activationDer = activateDer;
	   
	   for(int i = 1; i < dim.length; i++)
		   dim[i] = input * (1 - i / (dim.length - 1)) + output * i / (dim.length - 1);
	   
	   construct();
   }
   
   public NeuralNetwork(String fileName)
   {
	   try
	   {
		   derCounter = 0;
		   networkFront = new Node[1];
		   weightData = new ArrayList<double[]>();
		      
		   File file = new File(fileName);
		   Scanner scanner = new Scanner(file);
		   
		   String[] inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
		   dim = new int[inString.length];
		   
		   for(int i = 0; i < inString.length; i++)
			   dim[i] = Integer.parseInt(inString[i]);
		   
		   construct();
		   
		   for(double[] weight: weightData)
		   {
			   inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
			   
			   for(int i = 0; i < weight.length; i++)
				   weight[i] = Double.parseDouble(inString[i]);
		   }
	   }
	   catch(Exception e)
	   {
		   System.out.print(e);
	   }
   }
   
   public NeuralNetwork(String fileName, Activation activate, ActivationDer activateDer)
   {
	   try
	   {
		   derCounter = 0;
		   networkFront = new Node[1];
		   weightData = new ArrayList<double[]>();
		   activation = activate;
		   activationDer = activateDer;
		      
		   File file = new File(fileName);
		   Scanner scanner = new Scanner(file);
		   
		   String[] inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
		   dim = new int[inString.length];
		   
		   for(int i = 0; i < inString.length; i++)
			   dim[i] = Integer.parseInt(inString[i]);
		   
		   construct();
		   
		   for(double[] weight: weightData)
		   {
			   inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
			   
			   for(int i = 0; i < weight.length; i++)
				   weight[i] = Double.parseDouble(inString[i]);
		   }
	   }
	   catch(Exception e)
	   {
		   System.out.print(e);
	   }
   }
   
   public void construct()
   {
	   Node[] prevLayer = new Node[dim[0]];//create input layer node array using nodeFront as the prevLayer
	      
	      for(int i = 0; i < prevLayer.length; i++)
	         prevLayer[i] = new Node(networkFront, activation, activationDer);
	         
	      for(int i = 1; i < dim.length; i++)//loop through all the layers
	      {
	         Node[] currLayer = new Node[dim[i]];//create hidden layer 
	         
	         for(int j = 0; j < dim[i]; j++)
	            currLayer[j] = new Node(prevLayer, activation, activationDer);//link previous layer using 1 arg constructor
	            
	         prevLayer = currLayer;
	      }
	      
	      networkBack = new Node(prevLayer, activation, activationDer);//link neural networkBack to node networkBack
	      
	      networkBack.getPrevLayer()[0].saveWeightData(weightData, true, networkFront);
		   
		  for(int i = 1; i < networkBack.getPrevLayer().length; i++)
			   networkBack.getPrevLayer()[i].saveWeightData(weightData, false, networkFront);
   }
   
   public double[] calc(double[] input)
   {
      Node[] inputLayer = networkBack.getPrevLayer();
      
      while(!inputLayer[0].getPrevLayer().equals(networkFront))//get the inputLayer
         inputLayer = inputLayer[0].getPrevLayer();
      
      for(int i = 0; i < inputLayer.length; i++)//initialize networkBack
         inputLayer[i].setVal(input[i]);
         
      networkBack.getVal();//calculate networkBack
      
      double[] output = new double[networkBack.getPrevLayer().length];
      
      for(int i = 0; i < output.length; i++)
         output[i] = networkBack.getPrevLayer()[i].getVal();//set values for output
      
      return output;
   }
   
   public Node getNetworkBack()
   {
	   return networkBack;
   }
   
   public Node[] getNetworkFront()
   {
	   return networkFront;
   }
   public void clear()
   {
      networkBack.clear(networkFront);
   }
   
   public double backProp(double[] input, double[] expected)
   {
	  double error = 0; 
	  
      double[] output = calc(input);
      Node[] outputLayer = networkBack.getPrevLayer();
      
      for(int i = 0; i < outputLayer.length; i++)
      {
         outputLayer[i].setDer(output[i] - expected[i]);//sets derivative for output layer 
         error += 0.5 * Math.pow(output[i] - expected[i], 2);
      }
      
      networkBack.backProp(networkFront);
      clear();
      
      derCounter++;
      
      return error;
   }
   
   public double backPropPost(double[] output, double[] expected)
   {
	   double error = 0;
	   
	   Node[] outputLayer = networkBack.getPrevLayer();
	   
	   for(int i = 0; i < outputLayer.length; i++)
	   {
	      outputLayer[i].setDer(output[i] - expected[i]);//sets derivative for output layer 
	      error += 0.5 * Math.pow(output[i] - expected[i], 2);
	   }
	   
	   networkBack.backProp(networkFront);
	   clear();
	      
	   derCounter++;
	      
	   return error;
   }
   
   public void backPropGrad(double[] gradient)
   {
	   Node[] outputLayer = networkBack.getPrevLayer();
	   
	   for(int i = 0; i < outputLayer.length; i++)
	      outputLayer[i].setDer(gradient[i]);//sets derivative for output layer
	   
	   networkBack.backProp(networkFront);
	   clear();
	      
	   derCounter++;
   }
   
   public void updateWeight()
   {
      for(int i = 0; i < networkBack.getPrevLayer().length; i++)
         networkBack.getPrevLayer()[i].updateWeight(derCounter, rate);
      
      derCounter = 0;
   }
   
   public void setRate(double rate)
   {
	   this.rate = rate;
   }
   
   public int[] getDim()
   {
	   return dim;
   }
   
   public String saveNetwork(String fileName)
   {
	   try
	   {
		   File file = new File(fileName);
		   file.createNewFile();
		   
		   FileWriter fileWriter = new FileWriter(file.getAbsolutePath());
		   
		   fileWriter.write(Arrays.toString(dim) + "\n");
		   
		   for(double[] w: weightData)
			   fileWriter.write(Arrays.toString(w) + "\n");
		   
		   fileWriter.flush();
		   fileWriter.close();
		   
		   return file.getAbsolutePath();
	   }
	   catch(Exception e)
	   {
		   return e.toString();
	   }
   }
}