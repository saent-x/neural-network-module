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

}