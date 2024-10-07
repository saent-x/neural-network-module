package accuracy

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type BinaryAccuracy struct {
	Accuracy
}

func (binaryAccuracy *BinaryAccuracy) Init(y *mat.Dense, reinit bool) {

}

func (binaryAccuracy *BinaryAccuracy) Calculate(outputs *mat.Dense, y *mat.Dense) float64 {
	comparisons := binaryAccuracy.Compare(outputs, y)
	accuracy := stat.Mean(comparisons.RawMatrix().Data, nil)

	binaryAccuracy.AccumulatedSum += mat.Sum(comparisons)
	binaryAccuracy.AccumulatedCount += float64(comparisons.RawMatrix().Rows)

	return accuracy
}
func (binaryAccuracy *BinaryAccuracy) Compare(predictions *mat.Dense, y *mat.Dense) *mat.Dense {
	rows, cols := predictions.Dims()
	results := mat.NewDense(rows, cols, nil)

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
