package accuracy

import (
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Accuracy describes how often the largest confidence is the correct class in terms of fraction

func Calculate(outputs *mat.Dense, y *mat.Dense) float64 {
	// get index of max value in each row in softmax
	p_rows, _ := outputs.Dims()
	predictions := mat.NewDense(p_rows, 1, nil)

	lo.ForEach(lo.Range(p_rows), func(item int, index int) {
		row := outputs.RawRowView(index)
		predictions.SetRow(index, []float64{float64(lo.IndexOf(row, lo.Max(row)))})
	})

	// check if class-targets are one-hot encoded - convert them to sparse data
	c_rows, _ := y.Dims()
	if c_rows > 1 {
		OHE_class_targets := mat.NewDense(p_rows, 1, nil)

		lo.ForEach(lo.Range(c_rows), func(item int, index int) {
			var arr []float64
			row := y.RawRowView(index)
			OHE_class_targets.SetRow(index, append(arr, float64(lo.IndexOf(row, lo.Max(row)))))
		})
		y = mat.DenseCopyOf(OHE_class_targets)
	}

	pred_class_evaluation := mat.NewDense(p_rows, 1, nil)
	// assign 1 where predictions == y, then find the cumulative mean
	lo.ForEach(predictions.RawMatrix().Data, func(prediction float64, index int) {
		val := lo.Ternary[float64](prediction == y.RawMatrix().Data[index], 1, 0)
		pred_class_evaluation.SetRow(index, []float64{val})
	})

	accuracy := stat.Mean(pred_class_evaluation.RawMatrix().Data, nil)

	return accuracy
}
