package activation

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
	"math"
)

type Sigmoid struct {
	layer.LayerCommons
	layer.LayerNavigation
}

func (sigmoid *Sigmoid) Forward(inputs *mat.Dense, training bool) {
	sigmoid.Inputs = mat.DenseCopyOf(inputs)

	var output mat.Dense
	output.Apply(func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}, inputs)

	sigmoid.Output = mat.DenseCopyOf(&output)
}

func (sigmoid *Sigmoid) Backward(d_values *mat.Dense) {
	// 1 - sigmoid.Output
	var output_by_neg_output, one_neg_output, new_dinputs mat.Dense

	one_neg_output.Apply(func(i, j int, v float64) float64 {
		return 1 - v
	}, sigmoid.Output)
	output_by_neg_output.MulElem(&one_neg_output, sigmoid.Output)

	new_dinputs.MulElem(d_values, &output_by_neg_output)

	sigmoid.D_Inputs = mat.DenseCopyOf(&new_dinputs)
}

func (sigmoid *Sigmoid) Predictions(outputs *mat.Dense) *mat.Dense {
	rows, cols := outputs.Dims()
	result := mat.NewDense(rows, cols, nil)

	threshold := 0.5

	for i := 0; i < outputs.RawMatrix().Rows; i++ {
		for j := 0; j < outputs.RawMatrix().Cols; j++ {
			if outputs.At(i, j) > threshold {
				result.Set(i, j, 1.0)
			} else {
				result.Set(i, j, 0)
			}
		}
	}

	return result
}

func (sigmoid *Sigmoid) GetOutput() *mat.Dense { return sigmoid.Output }
