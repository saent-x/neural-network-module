using PandasNet;
using NumSharp;

namespace IDS_NN.core;

/**
	n_inputs: size of one record of the input data
	n_neurons: number of neurons
**/
public class LayerDense(int n_inputs, ushort n_neurons)
{
	/**
		multiply the _weights by the 0.10 to create a gaussian distribution bounded by zero
	**/
	private readonly NDArray _weights = 0.10 * np.random.randn(n_inputs, n_neurons);
	private readonly NDArray _biases = np.zeros((1, n_neurons));
	private NDArray? _output;
	public NDArray Output => _output ?? 0;
	
	
	public void Forward(NDArray inputs)
	{
		_output = np.dot(inputs, _weights) + _biases;
	}
}