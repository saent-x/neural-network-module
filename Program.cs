global using Spectre.Console;
global using Console = Spectre.Console.AnsiConsole;
using PandasNet;
using NumSharp;
using IDS_NN.core;


// set seed state
np.random.seed(0);

// var X = new float[3,4]{
// 	{1,2,3, 2.5f},
// 	{2.0f, 5.0f, -1.0f, 2.0f},
// 	{-1.5f, 2.7f, 3.3f, -0.8f}	
// };

// var layer_1 = new LayerDense(4, 5);
// var layer_2 = new LayerDense(5, 2);

// layer_1.Forward(X);
// //Console.WriteLine($"layer_1: [{layer_1.Output}]");

// layer_2.Forward(layer_1.Output);
// Console.WriteLine($"layer_2: [{layer_2.Output}]");

var inputs = new float[]{ 0,2, -1, 3.3f, -2.7f, 1.1f, 2.2f, -100 };
List<float> output = [];

foreach(var i in inputs) output.Add(Math.Max(0, i));

Console.Write("[ ");
output.ForEach(x => Console.Write($"{x}, "));
Console.Write(" ]\n");