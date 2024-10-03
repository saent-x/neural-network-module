package accuracy

import (
	"github.com/saent-x/ids-nn/core"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type CategoricalAccuracy struct {
	Accuracy
}

func (categoricalAccuracy *CategoricalAccuracy) Init(y *mat.Dense, reinit bool) {

}

func (categoricalAccuracy *CategoricalAccuracy) Calculate(outputs *mat.Dense, y *mat.Dense) float64 {
	comparisons := categoricalAccuracy.Compare(outputs, y)
	accuracy := stat.Mean(comparisons.RawMatrix().Data, nil)

	categoricalAccuracy.AccumulatedSum += mat.Sum(comparisons)
	categoricalAccuracy.AccumulatedCount += float64(comparisons.RawMatrix().Cols)

	return accuracy
}
func (categoricalAccuracy *CategoricalAccuracy) Compare(predictions *mat.Dense, y *mat.Dense) *mat.Dense {
	// get index of max value in each row in softmax
	rows, cols := predictions.Dims()

	// check if class-targets are one-hot encoded - convert them to sparse data
	y_rows, _ := y.Dims()

	if y_rows > 1 {
		y = mat.DenseCopyOf(core.OHEToSparse(y))
	}

	results := mat.NewDense(rows, cols, nil)
	// assign 1 where predictions == y, then find the cumulative mean

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if predictions.At(i, j) == y.At(i, j) {
				results.Set(i, j, 1)
			} else {
				results.Set(i, j, 0)
			}
		}
	}

	return results
}
