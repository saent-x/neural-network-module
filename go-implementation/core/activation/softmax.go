package activation

import (
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"math"
)

type SoftMax struct {
	Output   *mat.Dense
	D_Inputs *mat.Dense
}

func (sm *SoftMax) Forward(inputs *mat.Dense) {
	rows, columns := inputs.Dims()

	exp_values := mat.NewDense(rows, columns, nil)
	probabilities := mat.NewDense(rows, columns, nil)

	// find max in each row
	var max_in_rows []float64
	lo.ForEach(lo.Range(rows), func(item int, index int) {
		max_in_rows = append(max_in_rows, lo.Max(inputs.RawRowView(index)))
	})

	max_inputs := mat.NewDense(rows, 1, max_in_rows)
	sub_inputs := mat.NewDense(rows, columns, nil)

	// subtract col wise
	lo.ForEach(lo.Range(columns), func(item int, index int) {
		column := inputs.ColView(index)
		mx_column := max_inputs.ColView(0)
		sub := mat.NewVecDense(rows, nil)

		sub.SubVec(column, mx_column)
		sub_inputs.SetCol(index, sub.RawVector().Data)
	})

	// find exp of sub_inputs
	exp_values.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, sub_inputs)

	// find sum of each row in exp_values
	var sum_exp_values []float64
	lo.ForEach(lo.Range(rows), func(item int, index int) {
		sum_exp_values = append(sum_exp_values, lo.Sum(exp_values.RawRowView(index)))
	})
	sum_exp := mat.NewDense(rows, 1, sum_exp_values)

	// divide col wise
	lo.ForEach(lo.Range(columns), func(item int, index int) {
		sum_exp_column := sum_exp.ColView(0)
		exp_values_column := exp_values.ColView(index)
		div := mat.NewVecDense(rows, nil)

		div.DivElemVec(exp_values_column, sum_exp_column)
		probabilities.SetCol(index, div.RawVector().Data)
	})

	sm.Output = probabilities
}

func (sm *SoftMax) Backward(d_values *mat.Dense) {
	//r, c := inputs.Dims()
	sm.D_Inputs = mat.DenseCopyOf(d_values)
	sm.D_Inputs.Zero() // empty

	r, _ := d_values.Dims()

	for i, _ := range lo.Range(r) {
		raw_row := sm.Output.RawRowView(i)
		single_output := mat.NewDense(len(raw_row), 1, raw_row)

		_, c := single_output.T().Dims()
		diag_flat := mat.NewDiagDense(single_output.RawMatrix().Rows, raw_row)

		mul_single_output := mat.NewDense(single_output.RawMatrix().Rows, c, nil)
		mul_single_output.Mul(single_output, single_output.T())

		jacobian_matrix := mat.NewDense(single_output.RawMatrix().Rows, single_output.RawMatrix().Rows, nil)
		jacobian_matrix.Sub(diag_flat, mul_single_output)

		var result mat.Dense
		result.Mul(jacobian_matrix, d_values.RowView(i))

		sm.D_Inputs.SetRow(i, result.RawMatrix().Data)
	}
}
