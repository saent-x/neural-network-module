package activation

import (
	"gonum.org/v1/gonum/mat"
)

type ReLU struct {
	Output *mat.Dense
}

func (ar *ReLU) Forward(inputs *mat.Dense) {
	rows, columns := inputs.Dims()

	output := mat.NewDense(rows, columns, nil)
	output.Apply(func(i, j int, value float64) float64 {
		if value > 0 {
			return value
		}
		return 0
	}, inputs)

	ar.Output = output
}
