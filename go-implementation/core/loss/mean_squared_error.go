package loss

import (
	"github.com/saent-x/ids-nn/core"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

type MeanSquaredError struct {
	Loss
}

func (mse *MeanSquaredError) Calculate(output *mat.Dense, y *mat.Dense) (float64, float64) {
	sample_losses := mse.Forward(output, y)
	average_loss := stat.Mean(sample_losses.RawVector().Data, nil)

	return average_loss, mse.Regularization_Loss
}

func (mse *MeanSquaredError) Forward(y_true, y_pred *mat.Dense) *mat.VecDense {
	var fn mat.Dense

	fn.Sub(y_true, y_pred)
	fn.Apply(func(_, _ int, v float64) float64 {
		return math.Pow(v, 2)
	}, &fn)

	sample_losses := core.MeanOnLastAxis(&fn)

	return sample_losses
}

func (mse *MeanSquaredError) Backward(d_values, y_true *mat.Dense) {
	samples := d_values.RawMatrix().Rows
	outputs := len(d_values.RawRowView(0))

	var fn mat.Dense

	fn.Sub(y_true, d_values)
	fn.Apply(func(i, j int, v float64) float64 {
		result := (-2 * v) / float64(outputs)
		return result / float64(samples)
	}, &fn)

	mse.D_Inputs = &fn
}
