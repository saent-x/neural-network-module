package loss

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Loss function is used to quantify how wrong the model is.
// Categorical Cross Entropy is also one of the most commonly used loss functions with a softmax activation
// on the output layer. It works by comparing two probability distribution (predictions & targets)

type LossFunction struct {
	Loss float64
}

func (lf *LossFunction) Calc(softmax_output *mat.Dense, target_values *mat.Dense) {
	rows, columns := softmax_output.Dims()
	result := mat.NewDense(rows, columns, nil)

	result.Apply(func(i, j int, value float64) float64 {
		return math.Log(value) * target_values.At(i, j)
	}, softmax_output)

	// sum matrix all up
	lf.Loss = -mat.Sum(result)
}
