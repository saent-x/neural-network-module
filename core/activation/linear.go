package activation

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
)

type Linear struct {
	layer.LayerCommons
	layer.LayerNavigation
}

func (linear *Linear) Forward(inputs *mat.Dense, training bool) {
	linear.Inputs = inputs
	linear.Output = inputs
}

func (linear *Linear) Backward(d_values *mat.Dense) {
	linear.D_Inputs = mat.DenseCopyOf(d_values)
}

func (linear *Linear) GetOutput() *mat.Dense {
	return linear.Output
}

func (linear *Linear) Predictions(outputs *mat.Dense) *mat.Dense {
	return outputs
}
