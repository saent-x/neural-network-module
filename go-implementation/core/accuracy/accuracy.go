package accuracy

import (
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

// Accuracy describes how often the largest confidence is the correct class in terms of fraction
type IAccuracy interface {
	Init(y *mat.Dense, reinit bool)
	Compare(predictions, y *mat.Dense) *mat.Dense
}

type Accuracy struct {
}

func Calculate(accuracy_type IAccuracy, outputs *mat.Dense, y *mat.Dense) float64 {
	comparisons := accuracy_type.Compare(outputs, y)
	accuracy := stat.Mean(comparisons.RawMatrix().Data, nil)

	return accuracy
}

func BinaryCalculate(outputs *mat.Dense, y *mat.Dense, threshold float64) float64 {
	// Get the dimensions of the matrix
	rows, cols := outputs.Dims()

	// Create a new matrix to store the output (same dimensions as input)
	predictions := mat.NewDense(rows, cols, nil)

	// Loop through the matrix and apply thresholding
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if outputs.At(i, j) > threshold {
				predictions.Set(i, j, 1) // Set to 1 if the value is greater than the threshold
			} else {
				predictions.Set(i, j, 0) // Set to 0 otherwise
			}
		}
	}

	pred_class_evaluation := mat.NewDense(rows, 1, nil)
	// assign 1 where predictions == y, then find the cumulative mean
	for i, pred := range predictions.RawMatrix().Data {
		val := lo.Ternary[float64](pred == y.RawMatrix().Data[i], 1, 0)
		pred_class_evaluation.SetRow(i, []float64{val})
	}

	accuracy := stat.Mean(pred_class_evaluation.RawMatrix().Data, nil)

	return accuracy
}

// since we can't calvulate accuracy for regression problems, we can simulate it using a deviation from a ground truth value
func LinearCalculate(predictions, y *mat.Dense, accuracy_precision float64) float64 {
	var fn mat.Dense

	fn.Apply(func(i, j int, v float64) float64 {
		pred := predictions.At(i, j)
		abs_diff := math.Abs(pred - v)

		if abs_diff < accuracy_precision {
			return 1
		} else {
			return 0
		}
	}, y)

	return stat.Mean(fn.RawMatrix().Data, nil)
}
