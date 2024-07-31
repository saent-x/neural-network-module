namespace IDS_NN.pkg;

public class IDSNeuralNet{

    private Random _rand;
    public List<Layer> Layers { get; set; } = new();

    public IDSNeuralNet(double input_size, double output_size, double depth){
        // input layer of nn
        var input_layer = new Layer(input_size + 2);
        Layers.Add(input_layer);
        
        // middle layer of nn   
        for (var i = 0; i < depth; i++)
        {
            var middle_layer = new Layer(3);
            Layers.Add(middle_layer);
        }
        
        // output layer for nn
        var output_layer = new Layer(output_size);
        Layers.Add(output_layer);

        // Link all Layers
        for (int i = 1; i < Layers.Count; i++)
        {
            var previous_layer = Layers[i - 1];
            var next_layer = Layers[i];
            
            previous_layer.Link(next_layer);
        }
    }

    public List<double> GetResults(List<double> inputs)
    {
        // load inputs
        var first_layer = Layers[0];
        var local_inputs = new List<double>(inputs) { -1, 1 };

        for (var i = 0; i < local_inputs.Count; i++) first_layer.Nodes[i].Result = local_inputs[i];
        foreach (var layer in Layers) layer.Calc();

        var last_layer = Layers.Last();
        var results = last_layer.Nodes.Select(x => x.Result)
            .ToList();

        // clear results
        foreach (var node in Layers.SelectMany(layer => layer.Nodes)) node.Result = 0;

        return results;
    }

}