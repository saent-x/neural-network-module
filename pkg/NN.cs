namespace IDS_NN.pkg;

public class NN{

    private Random? _rand;
    private readonly List<Layer> _layers = [];

    public NN(double input_size, double output_size, double depth){
        // input layer of nn
        var input_layer = new Layer(input_size + 2);
        _layers.Add(input_layer);
        
        // middle layer of nn   
        for (var i = 0; i < depth; i++)
        {
            var middle_layer = new Layer(3);
            _layers.Add(middle_layer);
        }
        
        // output layer for nn
        var output_layer = new Layer(output_size);
        _layers.Add(output_layer);

        // Link all Layers
        for (var i = 1; i < _layers.Count; i++)
        {
            var previous_layer = _layers[i - 1];
            var next_layer = _layers[i];
            
            previous_layer.Link(next_layer);
        }
    }

    public List<double> GetResults(List<double> inputs)
    {
        // load inputs
        var first_layer = _layers[0];
        var local_inputs = new List<double>(inputs) { -1, 1 };
    
        for (var i = 0; i < local_inputs.Count; i++) first_layer.Nodes[i].Result = local_inputs[i];
        foreach (var layer in _layers) layer.Calc();

        var last_layer = _layers.Last();
        var results = last_layer.Nodes.Select(x => x.Result)
            .ToList();

        // clear results
        foreach (var node in _layers.SelectMany(layer => layer.Nodes)) node.Result = 0;

        return results;
    }   

    public void Train(List<DataFrame> training_data, double acceptable_score)
    {
        _rand = new Random(17);

        while (true)
        {
            var layers = _layers[0..];
            
            foreach(var layer in _layers)
                foreach(var node in layer.Nodes)
                foreach (var key in node.InputNodes.Keys)
                {
                    var original_score = GetScore(training_data);
                    var original_value = node.InputNodes[key];

                    node.InputNodes[key] += _rand.NextDouble() < 0.5 ? -1 * _rand.NextDouble() : _rand.NextDouble();
                    var new_score = GetScore(training_data);

                    if (new_score < original_score) Console.WriteLine("\t-> info: improved node");
                    else node.InputNodes[key] = original_value;
                }

            var score = GetScore(training_data);
            Console.WriteLine($"\t \t-* info: score {score}");

            if (!(score < acceptable_score)) continue;
            
            Console.WriteLine("\t-- info: training passed\n");
            return;
        }
    }
    
    private double GetScore(List<DataFrame> training_data)
    {
        var score = 0.0;

        foreach (var dataframe in training_data)
        {
            var results = GetResults(dataframe.Inputs!);
            for (int i = 0; i < results.Count; i++)
            {
                var target_output = dataframe.Targets![i];
                var actual_output = results[i];

                score += Math.Abs(target_output - actual_output);
            }
        }

        return score;
    }
}