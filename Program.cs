
using IDS_NN.pkg;

Console.WriteLine("info: loading training data...");
#region load training Data
List<DataFrame> training_data = [
    new DataFrame
    {
        Inputs = new List<double>() {0, 0},
        Targets = new List<double>() {0}
    },
    new DataFrame
    {
        Inputs = new List<double>() {0, 1},
        Targets = new List<double>() {1}
    },
    new DataFrame
    {
        Inputs = new List<double>() {1, 0},
        Targets = new List<double>() {1}
    },
    new DataFrame
    {
        Inputs = new List<double>() {0, 0},
        Targets = new List<double>() {0}
    }
];
#endregion

Console.WriteLine("info: generating neural network");
#region generate NN
var nn = new NN(2, 1, 2);
nn.Train(training_data, 0.05);
#endregion

Console.WriteLine("info: test results");
#region test and generate results
foreach (var data in training_data)
{
    var outputs = nn.GetResults(data.Inputs!);
    Console.WriteLine($"\nresults: I({data.Inputs![0]} , {data.Inputs[1]}) - O({outputs[0]})");
    Console.WriteLine($"expected result: O({data.Targets![0]})");
}

Console.WriteLine("complete: neural network operations complete.");
Console.ReadKey();
#endregion