package RecurrentNeuralNetwork;
import java.util.List;

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
   
   public Node(Node[] inLayer)//set values for the previous  layer and set random weights
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
   
   interface Activation
   {
	   double activate(double val);
	   
	   default Activation tanh()
	   {
		   return (val) -> (Math.exp(val) - Math.exp(val * -1)) / (Math.exp(val) + Math.exp(val * -1));
	   }
   }
   
   interface ActivationDer
   {
	   double activateDer(double val);
	   
	   default ActivationDer tanhDer()
	   {
		   return (val) -> 1 - val * val;
	   }
   }
}