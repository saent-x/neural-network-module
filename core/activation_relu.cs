using NumSharp;
// ReSharper disable All

namespace IDS_NN.core;

/*  Implementation of Rectified Linear Unit. */
public class ActivationReLU
{
	private NDArray? _output;
	public NDArray Output => _output ?? 0;
	
	public void Forward(NDArray n_inputs)
	{
		_output = np.maximum(0, n_inputs);
	}
}