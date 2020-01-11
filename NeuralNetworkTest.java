public class NeuralNetworkTest
{
   public static void main(String[] args)
   {
      int[] dim = {2,2,1};
      NeuralNetwork nn = new NeuralNetwork(dim);
      double[] input = {1.0,0.0};
      double[] output = {10};
      
      for(int i = 0; i < 1000; i++)
      {
      System.out.println(nn.calc(input)[0]);
      nn.clear();
      nn.backProp(input, output);
      nn.updateWeight();
      }
      
      
      
      
      
   }
}