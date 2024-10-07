package loss

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestCrossEntropyLossFunction_1(t *testing.T) {
	softmax_output := mat.NewDense(1, 3, []float64{0.7, 0.1, 0.2})
	target_output := mat.NewDense(1, 1, []float64{1})

	loss_function_1 := new(CategoricalCrossEntropy)
	result, _ := loss_function_1.Calculate(softmax_output, target_output, false)

	got := result
	want := 2.3025850929940455

	if got != want {
		t.Errorf("error: got %f | want %f", got, want)
	}
}

func TestCrossEntropyLossFunction_2(t *testing.T) {
	softmax_output := mat.NewDense(3, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08})
	target_output := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 1, 0})

	loss_function_1 := new(CategoricalCrossEntropy)
	result, _ := loss_function_1.Calculate(softmax_output, target_output, false)

	got := result
	want := 0.38506088005216804

	if got != want {
		t.Errorf("error: got %f | want %f", got, want)
	}
}
