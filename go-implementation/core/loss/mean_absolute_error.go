package loss

import (
	"github.com/saent-x/ids-nn/core"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

type MeanAbsoluteError struct {
	Loss
}

func (meanAbsoluteError *MeanAbsoluteError) Calculate(output *mat.Dense, y *mat.Dense, include_regularization bool) (float64, float64) {
	sample_losses := meanAbsoluteError.Forward(output, y)
	average_loss := stat.Mean(sample_losses.RawVector().Data, nil)

	meanAbsoluteError.AccumulatedSum += mat.Sum(sample_losses)
	meanAbsoluteError.AccumulatedCount += float64(sample_losses.Len())

	if !include_regularization {
		return average_loss, 0
	}

	return average_loss, meanAbsoluteError.CalcRegularizationLoss()
}

func (meanAbsoluteError *MeanAbsoluteError) Forward(y_true, y_pred *mat.Dense) *mat.VecDense {
	var fn mat.Dense

	fn.Sub(y_true, y_pred)
	fn.Apply(func(_, _ int, v float64) float64 {
		return math.Abs(v)
	}, &fn)

	sample_losses := core.MeanOnLastAxis(&fn)

	return sample_losses
}

func (meanAbsoluteError *MeanAbsoluteError) Backward(d_values, y_true *mat.Dense) {
	samples := d_values.RawMatrix().Rows
	outputs := len(d_values.RawRowView(0))

	var fn mat.Dense

	fn.Sub(y_true, d_values)
	fn.Apply(func(i, j int, v float64) float64 {
		return (core.Sign(v) / float64(outputs)) / float64(samples)
	}, &fn)

	meanAbsoluteError.D_Inputs = &fn
}
