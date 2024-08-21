global using Spectre.Console;
global using Console = Spectre.Console.AnsiConsole;
using PandasNet;
using NumSharp;
using IDS_NN.core;


// set seed state
np.random.seed(0);

var (X, y) = Utilities.SpiralData(100, 3);

//Console.WriteLine($"[{X.max()}]");
NDArray arr = new float[]{0.00129555f, 0.00139436f, 0.00293134f};
Console.WriteLine($"{np.maximum(0, arr)}");

// var layer_1 = new LayerDense(2, 5);
// var activation_1 = new ActivationReLU();
// //
// layer_1.Forward(X);
// activation_1.Forward(layer_1.Output);
// //
// Console.WriteLine($"{layer_1.Output.max()}");
// Console.WriteLine($"\n------------------------------------------\n");
// Console.WriteLine($"{activation_1.Output.max()}");

// ----------------------------------------------------
// var z1 = new double[3] { 1, 2, 3 };
// var z2 = new double[3] { 4, 5, 6 };
//
// var r1 = @Utilities.ConcatenateColwise(z1, z2);
//
// r1.PrintArray();
// Console.WriteLine($"[{r1.Shape}]");
