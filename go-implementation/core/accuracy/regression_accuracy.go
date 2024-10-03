package accuracy

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
)

type RegressionAccuracy struct {
	Precision float64

	Accuracy
}

func NewRegressionAccuracy() *RegressionAccuracy {
	return &RegressionAccuracy{Precision: 0}
}
func (regressionAccuracy *RegressionAccuracy) Init(y *mat.Dense, reinit bool) {
	if regressionAccuracy.Precision == 0 || reinit {
		regressionAccuracy.Precision = stat.StdDev(y.RawMatrix().Data, nil) / 250
	}
}

func (regressionAccuracy *RegressionAccuracy) Calculate(outputs *mat.Dense, y *mat.Dense) float64 {
	comparisons := regressionAccuracy.Compare(outputs, y)
	accuracy := stat.Mean(comparisons.RawMatrix().Data, nil)

	regressionAccuracy.AccumulatedSum += mat.Sum(comparisons)
	regressionAccuracy.AccumulatedCount += float64(comparisons.RawMatrix().Rows)

	return accuracy
}

func (regressionAccuracy *RegressionAccuracy) Compare(predictions, y *mat.Dense) *mat.Dense {
	result := mat.NewDense(y.RawMatrix().Rows, y.RawMatrix().Cols, nil)

	for i := 0; i < y.RawMatrix().Rows; i++ {
		for j := 0; j < y.RawMatrix().Cols; j++ {
			sub := math.Abs(predictions.At(i, j) - y.At(i, j))
			if sub < regressionAccuracy.Precision {
				result.Set(i, j, 1)
			} else {
				result.Set(i, j, 0)
			}
		}
	}

	return result
}
