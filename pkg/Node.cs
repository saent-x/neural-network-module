namespace IDS_NN.pkg;

public class Node{
    public Dictionary<Node, double> InputNodes { get; set; }
    public double Result { get; set; }
    public Func<double, double> Sigmoid { get; set; }

    public Node(Func<double, double> sigmoid, double result){
        InputNodes = new();

        Sigmoid = sigmoid;
        Result = result;
    }
        
    public void Add(Node node) =>
        InputNodes?.Add(node, 0.00001 /* TODO: update value*/);

    public void Calc(){
        foreach (var (node, weight) in InputNodes)
        {
            Result += node.Sigmoid(weight * node.Result);
            
            if (double.IsNaN(Result) || double.IsInfinity(Result) || double.IsNegativeInfinity(Result))
            {
                throw new Exception("error: bad result!");
            }
        }

        Result = Result / Math.Max(1, InputNodes.Count);
    }
}