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

func (meanSquaredError *MeanSquaredError) Calculate(output *mat.Dense, y *mat.Dense, include_regularization bool) (float64, float64) {
	sample_losses := meanSquaredError.Forward(output, y)
	average_loss := stat.Mean(sample_losses.RawVector().Data, nil)

	meanSquaredError.AccumulatedSum += mat.Sum(sample_losses)
	meanSquaredError.AccumulatedCount += float64(sample_losses.Len())

	if !include_regularization {
		return average_loss, 0
	}

	return average_loss, meanSquaredError.CalcRegularizationLoss()
}

func (meanSquaredError *MeanSquaredError) Forward(y_true, y_pred *mat.Dense) *mat.VecDense {
	var fn mat.Dense

	fn.Sub(y_true, y_pred)
	fn.Apply(func(_, _ int, v float64) float64 {
		return math.Pow(v, 2)
	}, &fn)

	sample_losses := core.MeanOnLastAxis(&fn)

	return sample_losses
}

func (meanSquaredError *MeanSquaredError) Backward(d_values, y_true *mat.Dense) {
	samples := d_values.RawMatrix().Rows
	outputs := len(d_values.RawRowView(0))

	var fn mat.Dense

	fn.Sub(y_true, d_values)
	fn.Apply(func(i, j int, v float64) float64 {
		result := (-2 * v) / float64(outputs)
		return result / float64(samples)
	}, &fn)

	meanSquaredError.D_Inputs = &fn
}
