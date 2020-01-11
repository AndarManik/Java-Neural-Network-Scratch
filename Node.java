public class Node
{
   private double val;
   private boolean isVal;
   
   private double der;
   private int derCounter;
   
   private Node[] prevLayer;
   
   private double[] weight;
   private double[] derWeight;
   
   private double bias;
   private double derBias;
   
   private final double RATE = 0.1;
   
   public Node(Node[] inLayer)//set values for the previous layer and set random weights
   {
      prevLayer = inLayer;
      weight = new double[prevLayer.length];
      derWeight = new double[prevLayer.length];
      derCounter = 0;
      derBias = 0;
      
      for(int i = 0; i < prevLayer.length; i++)
         weight[i] = Math.random();
         
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
   
   private double activation(double input)//reLu if it is less than 0 return 0 return x if else
   {
      return Math.max(0, input);
   }
   
   private double activationDer()
   {
      if(val == 0)
         return 0;
      return 1;
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
   
   public void setDer(double der)
   {
      this.der = der;
      
      for(int i = 0; i < derWeight.length; i++)
         derWeight[i] += prevLayer[i].getVal() * activationDer() * der;
         
      derBias += activation(val) / val * der;
      
      derCounter++;
      
   }
   
   public double[] getWeight()
   {
      return weight;
   }
   
   public double[] getDerWeight()
   {
      return derWeight;
   }
   
   //assumes prevLayers weights have been calculated. Calulates the derivatives for the nodes which are two layers behind it.
   // in doing so sets up the asumption for backProp on the next layer up. Stops recursion when the layer two layers behind it is the front of the network
   public void backProp(Node[] networkFront)
   {  
      Node[] twoLayerBack = prevLayer[0].getPrevLayer();
      
      if(twoLayerBack[0].getPrevLayer().equals(networkFront))
         return;
      
      for(int i = 0; i < twoLayerBack.length; i++)
      {
         double sumDer = 0;
         
         for(int j = 0; j < prevLayer.length; j++)//sum the derivatives of node
            sumDer += prevLayer[j].getDerWeight()[i] / twoLayerBack[i].getVal() * prevLayer[j].getWeight()[i];
           
         twoLayerBack[i].setDer(sumDer);
      }
      
      prevLayer[0].backProp(networkFront);//recur on layer down
   }
   
   public void updateWeight()
   {
      if(derCounter == 0)
         return;
         
      for(int i = 0; i < weight.length; i++)
      {
         weight[i] -= derWeight[i] * RATE / derCounter;
         prevLayer[i].updateWeight();
         derWeight[i] = 0;
      }
      
      derCounter = 0;
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