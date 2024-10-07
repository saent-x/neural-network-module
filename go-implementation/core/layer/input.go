package layer

import "gonum.org/v1/gonum/mat"

type InputLayer struct {
	LayerCommons
	LayerNavigation
}

func (inputLayer *InputLayer) Forward(inputs *mat.Dense, training bool) {
	inputLayer.Output = mat.DenseCopyOf(inputs)
}

// [Redundant function]: only exists to satisfy interface constraint
func (inputLayer *InputLayer) Backward(d_values *mat.Dense) {}
