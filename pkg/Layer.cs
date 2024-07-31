namespace IDS_NN.pkg;

public class Layer{
    public List<Node> Nodes { get; set; } = [];

    public Layer(double width) {
        for(var i = 0; i < width; i++) {
            var node = new Node(
                // Mathematical Sigmoid formula
                (double x) => (10 / (1 + Math.Pow(Math.E, -1 * x))) - 5,
                0
            );

            Nodes.Add(node);
        }
    }

    public void Calc(){
        foreach (var node in Nodes) node.Calc();
    }

    public void Link(Layer next_layer)  
    {
        foreach (var nextNode in next_layer.Nodes)
        foreach (var node in Nodes)
            nextNode.Add(node);
    }
}