package loss

import (
	"github.com/saent-x/ids-nn/core"
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
	Loss     float64
	D_Inputs *mat.Dense
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
		y_pred_y_true_product.MulElem(y_pred_clipped, y_true)

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

func (lf *CrossEntropyLossFunction) Backward(d_values *mat.Dense, y_true *mat.Dense) {
	r, c := d_values.Dims()
	r0, _ := y_true.Dims()

	samples := r
	labels := len(d_values.RawRowView(0))

	if r0 == 1 {
		y_true.CloneFrom(core.SparseToOHE(y_true, labels))
	}

	// negate y_true
	y_true.Apply(func(i, j int, v float64) float64 {
		return -v
	}, y_true)

	div_result := mat.NewDense(r, c, nil)
	div_result.DivElem(y_true, d_values) // only works if the shapes a,b are same

	div_result.Apply(func(i, j int, v float64) float64 {
		return v / float64(samples)
	}, div_result)

	lf.D_Inputs = div_result
}
