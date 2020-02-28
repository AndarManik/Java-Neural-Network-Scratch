import java.io.*;
import java.util.*;

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
      activation = (val) -> (Math.exp(val) - Math.exp(val * -1)) / (Math.exp(val) + Math.exp(val * -1));
      activationDer = (val) -> 1 - val * val;
      
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
	   activation = (val) -> (Math.exp(val) - Math.exp(val * -1)) / (Math.exp(val) + Math.exp(val * -1));
	      activationDer = (val) -> 1 - val * val;
		
	   for(int i = 0; i < dim.length; i++)
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
	   
	   for(int i = 0; i < dim.length; i++)
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
		   activation = (val) -> (Math.exp(val) - Math.exp(val * -1)) / (Math.exp(val) + Math.exp(val * -1));
		   activationDer = (val) -> 1 - val * val;
		      
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
   
   public class Node
   {
   	private Node[] prevLayer;
   	   
   	private double val;
   	private boolean isVal;
   	   
   	private double der;
   	private boolean isDer;
   	   
   	private double[] weight;//last weight is bias
   	private double[] derWeight;
   	   
   	private Activation activation;
   	private ActivationDer activationDer;
      
      public Node(Node[] inLayer)//set values for the previous layer and set random weights
      {
         prevLayer = inLayer;
         weight = new double[prevLayer.length + 1];
         derWeight = new double[prevLayer.length + 1];
         
         for(int i = 0; i < weight.length; i++)
            weight[i] = Math.random() * 2 - 1;
            
         weight[weight.length - 1] = Math.random();
      }
      
      public Node(Node[] inLayer, Activation activate, ActivationDer activateDer)//set values for the previous  layer and set random weights
      {
         prevLayer = inLayer;
         weight = new double[prevLayer.length + 1];
         derWeight = new double[prevLayer.length + 1];
         
         for(int i = 0; i < weight.length; i++)
            weight[i] = Math.random() * 2 - 1;
            
         weight[weight.length - 1] = Math.random();
         
         activation = activate;
         activationDer = activateDer;
      }
      
      public double getVal()
      {
         if(isVal)
            return val;
            
         double sum = 0;
         
         for(int i = 0; i < prevLayer.length; i++)
            sum += prevLayer[i].getVal() * weight[i];
         
         val = activation.activate(sum + weight[weight.length -1]);
         isVal = true;
         
         return val;
      }
      
      public void setVal(double input)
      {
         val = input;
         isVal = true;
      }
      
      public Node[] getPrevLayer()
      {
         return prevLayer;
      }
      
      public double getDer() 
      {
   	   return der;   
      }
      
      public void setDer(double der)
      {
   	  this.der = der;
         
         for(int i = 0; i < prevLayer.length; i++)
            derWeight[i] += prevLayer[i].getVal() * activationDer.activateDer(val) * der;
            
         derWeight[derWeight.length - 1] += activationDer.activateDer(val) * der;
         
         isDer = true;
      }
      
      public double[] getWeight()
      {
         return weight;
      }
      
      public double[] getDerWeight()
      {
         return derWeight;
      }
      
      public void saveWeightData(List<double[]> weightData, boolean recur, Node[] networkFront)
      {
   	   if(recur && !prevLayer[0].prevLayer.equals(networkFront))
   	   {
   		   prevLayer[0].saveWeightData(weightData, true, networkFront);
   		   
   		   for(int i = 1; i < prevLayer.length; i++)
   			   prevLayer[i].saveWeightData(weightData, false, networkFront);
   	   }
   	   
   	   weightData.add(weight);
      }
      
      //assumes prevLayers weights have been calculated. Calculates the derivatives for the nodes which are two layers behind it.
      // in doing so sets up the assumption for backProp on the next layer up. Stops recursion when the layer two layers behind it is the front of the network
      public void backProp(Node[] networkFront)
      {  
         Node[] twoLayerBack = prevLayer[0].getPrevLayer();
         
         if(twoLayerBack[0].getPrevLayer().equals(networkFront))
            return;
         
         for(int i = 0; i < twoLayerBack.length; i++)
         {
            double sumDer = 0;
            
            for(int j = 0; j < prevLayer.length; j++)//sum the derivatives of node
           		 sumDer += prevLayer[j].activationDer.activateDer(val) * prevLayer[j].getDer() * prevLayer[j].getWeight()[i];
              
            twoLayerBack[i].setDer(sumDer);
         }
         
         prevLayer[0].backProp(networkFront);//recur on layer down
      }
      
      public void updateWeight(int derCounter, double rate)
      {
         if(isDer == false)
            return;
            
         for(int i = 0; i < prevLayer.length; i++)
         {
            weight[i] -= derWeight[i] * rate / derCounter;
            prevLayer[i].updateWeight(derCounter, rate);
            derWeight[i] = 0;
         }
         
         weight[weight.length - 1] -= derWeight[derWeight.length - 1] * rate / derCounter;
         derWeight[derWeight.length - 1] = 0;
         isDer = false;
      }
      
      public void clear(Node[] networkFront)
      {
         if(!isVal)
            return;
         
         val = 0;
         isVal = false;
         
         if(!prevLayer.equals(networkFront))
         {
            for(int i = 0; i < prevLayer.length; i++)
               prevLayer[i].clear(networkFront);
         }
      }
   }
   interface Activation
   {
	   double activate(double val);
   }
   
   interface ActivationDer
   {
	   double activateDer(double val);
   }
}

