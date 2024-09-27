package loss

import (
	"github.com/saent-x/ids-nn/core"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

type BinaryCrossEntropyLoss struct {
	//Regularization_Loss float64
	Loss
}

func (b *BinaryCrossEntropyLoss) Calculate(output *mat.Dense, y *mat.Dense) (float64, float64) {
	sample_losses := b.Forward(output, y)
	average_loss := stat.Mean(sample_losses.RawVector().Data, nil)

	return average_loss, b.Regularization_Loss
}

func (b *BinaryCrossEntropyLoss) Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.VecDense {
	var y_pred_clipped mat.Dense
	y_pred_clipped.Apply(func(i, j int, value float64) float64 {
		return lo.Clamp(value, 1e-7, 1-1e-7)
	}, y_pred)

	// -(y_true * np.log(y_pred_clipped))
	var fn_1, fn_2, fn_3, eqn_1, eqn_2, result mat.Dense

	fn_1.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, &y_pred_clipped)
	fn_2.Apply(func(i, j int, v float64) float64 {
		return math.Log(1 - v)
	}, &y_pred_clipped)
	fn_3.Apply(func(i, j int, v float64) float64 {
		return 1 - v
	}, y_true)

	eqn_1.MulElem(y_true, &fn_1) // TODO: debug here when epoch is 10000
	eqn_2.MulElem(&fn_3, &fn_2)

	result.Add(&eqn_1, &eqn_2)
	result.Apply(func(i, j int, v float64) float64 {
		return -v
	}, &result)

	sample_losses := core.MeanOnLastAxis(&result)

	return sample_losses
}

func (b *BinaryCrossEntropyLoss) Backward(d_values *mat.Dense, y_true *mat.Dense) {
	samples := d_values.RawMatrix().Rows
	outputs := len(d_values.RawRowView(0))

	var clipped_d_values, one_neg_ytrue, one_neg_clipped_dvalues, y_true_div_clipped_values, one_neg_y_true_clipped_dvalues, new_dinputs mat.Dense
	clipped_d_values.Apply(func(i, j int, value float64) float64 {
		return lo.Clamp(value, 1e-7, 1-1e-7)
	}, d_values)

	one_neg_ytrue.Apply(func(i, j int, v float64) float64 {
		return 1 - v
	}, y_true)
	one_neg_clipped_dvalues.Apply(func(i, j int, v float64) float64 {
		return 1 - v
	}, &clipped_d_values)

	y_true_div_clipped_values.DivElem(y_true, &clipped_d_values)
	one_neg_y_true_clipped_dvalues.DivElem(&one_neg_ytrue, &one_neg_clipped_dvalues)

	new_dinputs.Sub(&y_true_div_clipped_values, &one_neg_y_true_clipped_dvalues)

	new_dinputs.Apply(func(i, j int, v float64) float64 {
		result := -v / float64(outputs)
		return result / float64(samples)
	}, &new_dinputs)

	b.D_Inputs = mat.DenseCopyOf(&new_dinputs)
}
