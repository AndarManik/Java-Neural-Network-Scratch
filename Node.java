public class Node
{
   private Node[] prevLayer;
   
   private double val;
   private boolean isVal;
   
   private double der;
   private boolean isDer;
   
   private double[] weight;
   private double[] derWeight;
   
   private double bias;
   private double derBias;
   
   public Node(Node[] inLayer)//set values for the previous layer and set random weights
   {
      prevLayer = inLayer;
      weight = new double[prevLayer.length];
      derWeight = new double[prevLayer.length];
      derBias = 0;
      
      for(int i = 0; i < prevLayer.length; i++)
         weight[i] = Math.random() * 2 - 1;
         
      bias = Math.random();
   }
   
   public double getVal()
   {
      if(isVal)
         return val;
         
      double sum = 0;
      
      for(int i = 0; i < prevLayer.length; i++)
         sum += prevLayer[i].getVal() * weight[i];
      
      val = activation(sum + bias);
      isVal = true;
      
      return val;
   }
   
   private double activation(double input)//hyperbolic tangent
   {
      return (Math.exp(input) - Math.exp(input * -1)) / (Math.exp(input) + Math.exp(input * -1));
   }
   
   private double activationDer()
   {
      return 1 - val * val;
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
      
      for(int i = 0; i < derWeight.length; i++)
         derWeight[i] += prevLayer[i].getVal() * activationDer() * der;
         
      derBias += activationDer() * der;
      
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
        		 sumDer += prevLayer[j].activationDer() * prevLayer[j].getDer() * prevLayer[j].getWeight()[i];
           
         twoLayerBack[i].setDer(sumDer);
      }
      
      prevLayer[0].backProp(networkFront);//recur on layer down
   }
   
   public void updateWeight(int derCounter, double rate)
   {
      if(isDer == false)
         return;
         
      for(int i = 0; i < weight.length; i++)
      {
         weight[i] -= derWeight[i] * rate / derCounter;
         prevLayer[i].updateWeight(derCounter, rate);
         derWeight[i] = 0;
      }
      
      bias -= derBias * rate / derCounter;
      derBias = 0;
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