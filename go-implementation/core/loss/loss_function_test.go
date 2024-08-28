package loss

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestLossFunction(t *testing.T) {
	softmax_output := mat.NewDense(1, 3, []float64{0.7, 0.1, 0.2})
	target_output := mat.NewDense(1, 3, []float64{1, 0, 0})

	loss_function_1 := new(LossFunction)

	loss_function_1.Calc(softmax_output, target_output)

	got := loss_function_1.Loss
	want := 0.35667494393873245

	if got != want {
		t.Errorf("error: got %f | want %f", got, want)
	}
}
