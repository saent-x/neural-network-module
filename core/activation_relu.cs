using NumSharp;

namespace IDS_NN.core;

/*  Implementation of Rectified Linear Unit. */
public class ActivationReLU(int n_inputs)
{
	private NDArray? _output;
	public NDArray Output => _output ?? 0;
	public void Forward()
	{
		_output = np.maximum(0, n_inputs);
	}
}