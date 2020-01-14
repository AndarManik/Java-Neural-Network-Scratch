
public class NeuralNetwork
{
   private Node networkBack;
   private Node[] networkFront;
   private int[] dim;
   
   public NeuralNetwork(int[] dim)//creates networkBack with dimensions;
   {
      this.dim = dim;
      networkFront = new Node[1];
      Node[] prevLayer = new Node[dim[0]];//create input layer node array using nodeFront as the prevLayer
      
      for(int i = 0; i < prevLayer.length; i++)
         prevLayer[i] = new Node(networkFront);
         
      for(int i = 1; i < dim.length; i++)//loop through all the layers
      {
         Node[] currLayer = new Node[dim[i]];//create hidden layer 
         
         for(int j = 0; j < dim[i]; j++)
            currLayer[j] = new Node(prevLayer);//link previous layer using 1 arg constructor
            
         prevLayer = currLayer;
      }
      networkBack = new Node(prevLayer);//link neural networkBack to node networkBack
   }
   
   public double[] calc(double[] input)
   {
      Node[] inputLayer = networkBack.getPrevLayer();
      
      while(!inputLayer[0].getPrevLayer().equals(networkFront))//get the inputLayer
         inputLayer = inputLayer[0].getPrevLayer();
      
      for(int i = 0; i < inputLayer.length; i++)//initialize networkBack
         inputLayer[i].setVal(input[i]);
         
      networkBack.getVal();//calc networkBack
      
      double[] output = new double[networkBack.getPrevLayer().length];
      
      for(int i = 0; i < output.length; i++)
         output[i] = networkBack.getPrevLayer()[i].getVal();//set values for output
      
      return output;
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
      
      return error;
   }
   
   public void updateWeight()
   {
      for(int i = 0; i < networkBack.getPrevLayer().length; i++)
         networkBack.getPrevLayer()[i].updateWeight();
   }
}