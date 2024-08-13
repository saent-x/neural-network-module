global using Spectre.Console;
global using Console = Spectre.Console.AnsiConsole;
using PandasNet;
using NumSharp;
using IDS_NN.core;

// set seed state
np.random.seed(0);

var X = new float[3,4]{
	{1,2,3, 2.5f},
	{2.0f, 5.0f, -1.0f, 2.0f},
	{-1.5f, 2.7f, 3.3f, -0.8f}
};

var layer_1 = new LayerDense(4, 5);
var layer_2 = new LayerDense(5, 2);

layer_1.Forward(X);