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
func (ra *RegressionAccuracy) Init(y *mat.Dense, reinit bool) {
	if ra.Precision == 0 || reinit {
		ra.Precision = stat.StdDev(y.RawMatrix().Data, nil) / 250
	}
}

func (ra *RegressionAccuracy) Compare(predictions, y *mat.Dense) *mat.Dense {
	result := mat.NewDense(y.RawMatrix().Rows, y.RawMatrix().Cols, nil)

	for i := 0; i < y.RawMatrix().Rows; i++ {
		for j := 0; j < y.RawMatrix().Cols; j++ {
			sub := math.Abs(predictions.At(i, j) - y.At(i, j))
			if sub < ra.Precision {
				result.Set(i, j, 1)
			} else {
				result.Set(i, j, 0)
			}
		}
	}

	return result
}
