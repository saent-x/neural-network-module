package activation

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
)

type ReLU struct {
	layer.LayerCommons
	layer.LayerNavigation
}

func (relu *ReLU) Forward(inputs *mat.Dense, training bool) {
	relu.Inputs = mat.DenseCopyOf(inputs) // set inputs to be used for backpropagation

	var output mat.Dense
	output.Apply(func(i, j int, value float64) float64 {
		if value > 0 {
			return value
		}
		return 0
	}, inputs)

	relu.Output = mat.DenseCopyOf(&output)
}

func (relu *ReLU) Backward(d_values *mat.Dense) {
	relu.D_Inputs = mat.DenseCopyOf(d_values)
	relu.D_Inputs.Apply(func(i, j int, value float64) float64 {
		if relu.Inputs.At(i, j) <= 0 {
			return 0
		}
		return value
	}, relu.D_Inputs)
}

func (relu *ReLU) GetOutput() *mat.Dense {
	return relu.Output
}
