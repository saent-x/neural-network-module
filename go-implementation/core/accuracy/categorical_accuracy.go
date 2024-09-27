package accuracy

import (
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
)

type CategoricalAccuracy struct {
}

func (ra *CategoricalAccuracy) Init(y *mat.Dense, reinit bool) {

}
func (ca *CategoricalAccuracy) Compare(outputs *mat.Dense, y *mat.Dense) *mat.Dense {
	// get index of max value in each row in softmax
	p_rows, _ := outputs.Dims()
	predictions := mat.NewDense(p_rows, 1, nil)

	for i := 0; i < p_rows; i++ {
		row := outputs.RawRowView(i)
		predictions.SetRow(i, []float64{float64(lo.IndexOf(row, lo.Max(row)))})
	}

	// check if class-targets are one-hot encoded - convert them to sparse data
	c_rows, _ := y.Dims()
	if c_rows > 1 {
		OHE_class_targets := mat.NewDense(p_rows, 1, nil)

		for i := 0; i < c_rows; i++ {
			var arr []float64
			row := y.RawRowView(i)
			OHE_class_targets.SetRow(i, append(arr, float64(lo.IndexOf(row, lo.Max(row)))))
		}
		y = mat.DenseCopyOf(OHE_class_targets)
	}

	pred_class_evaluation := mat.NewDense(p_rows, 1, nil)
	// assign 1 where predictions == y, then find the cumulative mean
	for i, pred := range predictions.RawMatrix().Data {
		val := lo.Ternary[float64](pred == y.RawMatrix().Data[i], 1, 0)
		pred_class_evaluation.SetRow(i, []float64{val})
	}

	return pred_class_evaluation
}
