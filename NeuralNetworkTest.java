public class NeuralNetworkTest
{
   public static void main(String[] args)
   {
      int[] dim = {2,2,1};
      NeuralNetwork nn = new NeuralNetwork(dim);
      double[] input1 = {0,0};
      double[] output1 = {0};
      
      double[] input3 = {1,0};
      double[] output3 = {1};
      
      double[] input2 = {0,1};
      double[] output2 = {1};
      
      double[] input4 = {1,1};
      double[] output4 = {0};
      
      double curError;
      
      
      
      for(int i = 0; i < 10000000; i++)
      {
    	  curError = 0;
    	  
    	  
    	  System.out.println("00 " + nn.calc(input1)[0]);
    	  nn.clear();
    	  curError += nn.backProp(input1, output1);

      
    	  System.out.println("01 " + nn.calc(input2)[0]);
    	  nn.clear();
    	  curError += nn.backProp(input2, output2);

      
    	  System.out.println("10 " + nn.calc(input3)[0]);
    	  nn.clear();
    	  curError += nn.backProp(input3, output3);

          
    	  System.out.println("11 " + nn.calc(input4)[0]);
    	  nn.clear();
    	  curError += nn.backProp(input4, output4);
    	  System.out.println(curError);
    	  
    	  nn.updateWeight();
      }
      
      
      
      
      
   }
}