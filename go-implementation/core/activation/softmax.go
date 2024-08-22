package activation

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type SoftMax struct {
	Output *mat.Dense
}

func (sm *SoftMax) Forward(inputs *mat.Dense) {
	rows, columns := inputs.Dims()

	// get exponential of input values
	exp_values := mat.NewDense(rows, columns, nil)
	norm_values := mat.NewDense(rows, columns, nil)

	sum_exp_values := 0.0 // Norm base
	exp_values.Apply(func(i, j int, value float64) float64 {
		result := math.Exp(value)
		sum_exp_values += result

		return result
	}, inputs)

	// normalize exp values
	sum_norm_values := 0.0
	norm_values.Apply(func(i, j int, value float64) float64 {
		normalized_value := value / sum_exp_values
		sum_norm_values += normalized_value

		return normalized_value
	}, exp_values)

	sm.Output = norm_values
}
