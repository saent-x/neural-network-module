package loss

import (
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"log"
	"math"
)

// Loss function is used to quantify how wrong the model is.
// Categorical Cross Entropy is also one of the most commonly used loss functions with a softmax activation
// on the output layer. It works by comparing two probability distribution (predictions & targets)
type ILoss interface {
	Calculate(output *mat.Dense, y *mat.Dense) float64
	forward(output *mat.Dense, y *mat.Dense)
}

type CrossEntropyLossFunction struct {
	Loss float64
}

func (lf *CrossEntropyLossFunction) Calculate(output *mat.Dense, y *mat.Dense) float64 {
	sample_losses := lf.forward(output, y)
	average_loss := stat.Mean(sample_losses, nil)

	return average_loss
}

func (lf *CrossEntropyLossFunction) forward(y_pred *mat.Dense, y_true *mat.Dense) []float64 {
	samples, columns := y_pred.Dims()
	rows, _ := y_true.Dims()

	y_pred_clipped := mat.NewDense(samples, columns, nil)
	y_pred_clipped.Apply(func(i, j int, value float64) float64 {
		return lo.Clamp(value, 1e-7, 1-1e-7)
	}, y_pred)

	if rows == 1 {
		// check this calculation
		var correct_confidences []float64

		lo.ForEach(lo.Range(samples), func(item int, index int) {
			value := y_pred_clipped.At(index, int(y_true.RawMatrix().Data[index]))
			correct_confidences = append(correct_confidences, -math.Log(value))
		})

		return correct_confidences
	} else if rows > 1 {
		// for hot-one encoded categorical variables
		var correct_confidences []float64

		y_pred_y_true_product := mat.NewDense(samples, columns, nil)
		y_pred_y_true_product.Mul(y_pred_clipped, y_true)

		// sum each row and calc the natural logarithm of the sum
		lo.ForEach(lo.Range(samples), func(item int, index int) {
			row := y_pred_y_true_product.RawRowView(index)
			correct_confidences = append(correct_confidences, -math.Log(lo.Sum(row)))
		})

		return correct_confidences
	} else {
		log.Fatalln("invalid row/sample size")
		return nil
	}
}
