package activation

import (
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
)

type SoftMax struct {
	layer.LayerCommons
	layer.LayerNavigation
}

func (softmax *SoftMax) Forward(inputs *mat.Dense, training bool) {
	softmax.Inputs = inputs

	rows, columns := inputs.Dims()

	var exp_values mat.Dense

	// find max in each row
	var max_in_rows []float64
	for i := 0; i < rows; i++ {
		max_in_rows = append(max_in_rows, lo.Max(inputs.RawRowView(i)))
	}

	max_inputs := mat.NewDense(rows, 1, max_in_rows)
	sub_inputs := mat.NewDense(rows, columns, nil)

	sub_inputs = mat.DenseCopyOf(core.SubtractUnevenMatrices(inputs, max_inputs))

	// find exp of sub_inputs
	exp_values.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, sub_inputs)

	// find sum of each row in exp_values
	var sum_exp_values []float64
	for i := 0; i < rows; i++ {
		sum_exp_values = append(sum_exp_values, lo.Sum(exp_values.RawRowView(i)))
	}

	sum_exp := mat.NewDense(rows, 1, sum_exp_values)

	result := core.DivideUnevenMatrices(&exp_values, sum_exp)

	softmax.Output = mat.DenseCopyOf(result)
}

func (softmax *SoftMax) Backward(d_values *mat.Dense) {
	softmax.D_Inputs = mat.DenseCopyOf(d_values)
	softmax.D_Inputs.Zero() // empty

	r, _ := d_values.Dims()

	for i := 0; i < r; i++ {
		raw_row := softmax.Output.RawRowView(i)

		single_output := mat.NewDense(len(raw_row), 1, raw_row)
		diag_flat := mat.NewDiagDense(single_output.RawMatrix().Rows, raw_row)

		var mul_single_output, jacobian_matrix, result mat.Dense

		mul_single_output.Mul(single_output, single_output.T())
		jacobian_matrix.Sub(diag_flat, &mul_single_output)
		result.Mul(&jacobian_matrix, d_values.RowView(i))

		softmax.D_Inputs.SetRow(i, result.RawMatrix().Data)
	}
}

func (softmax *SoftMax) Predictions(outputs *mat.Dense) *mat.Dense {
	rows := outputs.RawMatrix().Rows
	argmax := mat.NewDense(1, rows, nil)

	for i := 0; i < rows; i++ {
		max_in_row := floats.MaxIdx(outputs.RawRowView(i))
		argmax.Set(0, i, float64(max_in_row))
	}

	return argmax
}

func (softmax *SoftMax) GetOutput() *mat.Dense {
	return softmax.Output
}
