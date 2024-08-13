using PandasNet;
using NumSharp;
using Serilog.Debugging;

namespace IDS_NN.core;

/**
	n_inputs: size of one record of the input data
	n_neurons: number of neurons
**/
public class LayerDense(int n_inputs, int n_neurons)
{
	/**
		multiply the _weights by the 0.10 to create a gaussian distribution bounded by zero
	**/
	private NDArray _weights = 0.10 * np.random.randn(n_inputs, n_neurons);
	private NDArray _biases = np.zeros((1, n_neurons));
	private NDArray? _output;
	
	public void Forward(float[,] inputs)
	{
		_output = np.dot(inputs, _weights) + _biases;
	}
}