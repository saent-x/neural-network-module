package activation

import (
	"gonum.org/v1/gonum/mat"
)

type ReLU struct {
	Inputs   *mat.Dense
	Output   *mat.Dense
	D_Inputs *mat.Dense
}

func (self *ReLU) Forward(inputs *mat.Dense) {
	self.Inputs = mat.DenseCopyOf(inputs) // set inputs to be used for backpropagation
	rows, columns := inputs.Dims()

	output := mat.NewDense(rows, columns, nil)
	output.Apply(func(i, j int, value float64) float64 {
		if value > 0 {
			return value
		}
		return 0
	}, inputs)

	self.Output = mat.DenseCopyOf(output)
}

func (self *ReLU) Backward(d_values *mat.Dense) {
	self.D_Inputs = mat.DenseCopyOf(d_values)
	self.D_Inputs.Apply(func(i, j int, value float64) float64 {
		if self.Inputs.At(i, j) <= 0 {
			return 0
		}
		return value
	}, self.D_Inputs)
}
