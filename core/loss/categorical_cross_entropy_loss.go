package loss

import (
	"github.com/saent-x/ids-nn/core"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

// LossValue function is used to quantify how wrong the model is.
// Categorical Cross Entropy is also one of the most commonly used loss functions with a softmax activation
// on the output layer. It works by comparing two probability distribution (predictions & targets)

type CategoricalCrossEntropy struct {
	LossValue float64
	Loss
}

func (categoricalCrossEntropy *CategoricalCrossEntropy) Calculate(output *mat.Dense, y *mat.Dense, include_regularization bool) (float64, float64) {
	sample_losses := categoricalCrossEntropy.Forward(output, y)
	average_loss := stat.Mean(sample_losses.RawVector().Data, nil)

	categoricalCrossEntropy.AccumulatedSum += mat.Sum(sample_losses)
	categoricalCrossEntropy.AccumulatedCount += float64(sample_losses.Len())

	if !include_regularization {
		return average_loss, 0
	}

	return average_loss, categoricalCrossEntropy.CalcRegularizationLoss()
}

func (categoricalCrossEntropy *CategoricalCrossEntropy) Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.VecDense {
	samples, _ := y_pred.Dims()
	rows, cols := y_true.Dims()

	var y_pred_clipped mat.Dense
	y_pred_clipped.Apply(func(i, j int, value float64) float64 {
		return lo.Clamp(value, 1e-7, 1-1e-7)
	}, y_pred)

	if rows == 1 || cols == 1 {
		// check this calculation
		correct_confidences := mat.NewVecDense(samples, nil)
		for i := 0; i < samples; i++ {
			value := y_pred_clipped.At(i, int(y_true.RawMatrix().Data[i]))
			correct_confidences.SetVec(i, -math.Log(value)) //= append(correct_confidences, -math.Log(value))
		}

		return correct_confidences
	} else {
		// for hot-one encoded categorical variables
		correct_confidences := mat.NewVecDense(samples, nil)

		var y_pred_y_true_product mat.Dense
		y_pred_y_true_product.MulElem(&y_pred_clipped, y_true)

		// sum each row and calc the natural logarithm of the sum
		for i := 0; i < samples; i++ {
			row := y_pred_y_true_product.RawRowView(i)
			correct_confidences.SetVec(i, -math.Log(lo.Sum(row))) //= append(correct_confidences, -math.Log(lo.Sum(row)))
		}

		return correct_confidences
	}

	return nil
}

func (categoricalCrossEntropy *CategoricalCrossEntropy) Backward(d_values *mat.Dense, y_true *mat.Dense) {
	samples := d_values.RawMatrix().Rows
	labels := len(d_values.RawRowView(0))

	if y_true.RawMatrix().Rows == 1 {
		y_true = core.SparseToOHE(y_true, labels)
	}

	// negate y_true
	y_true.Apply(func(i, j int, v float64) float64 {
		return -v
	}, y_true)

	var div_result mat.Dense
	div_result.DivElem(y_true, d_values) // only works if the shapes a,b are same
	div_result.Apply(func(i, j int, v float64) float64 {
		return v / float64(samples)
	}, &div_result)

	categoricalCrossEntropy.D_Inputs = &div_result
}
