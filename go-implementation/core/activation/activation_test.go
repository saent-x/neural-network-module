package activation

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestActivationCreation(t *testing.T) {
	activation_1 := new(ReLU)
	activation_2 := new(SoftMax)

	if activation_1 == nil || activation_2 == nil {
		t.Errorf("error: activation_1 & activation_2 are nil!")
	}
}

func TestActivationForwardFunction(t *testing.T) {
	X, _ := core.SpiralData(100, 3)

	layer_1 := layer.CreateLayer(2, 3, 0, 0, 0, 0)
	layer_2 := layer.CreateLayer(3, 3, 0, 0, 0, 0)

	activation_1 := new(ReLU)
	activation_2 := new(SoftMax)

	layer_1.Forward(X, true)
	activation_1.Forward(layer_1.Output, true)
	layer_2.Forward(activation_1.Output, true)
	activation_2.Forward(layer_2.Output, true)

	if activation_2.Output == nil {
		t.Errorf("error: activation_2 is nil and invalid!")
	}
}

func TestSoftmaxFunction(t *testing.T) {
	inputs := mat.NewDense(3, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08})

	softmax := new(SoftMax)
	softmax.Forward(inputs, true)

	fmt.Println(mat.Formatted(softmax.Output))

	if softmax.Output == nil {
		t.Errorf("error: softmax output is nil!")
	}
}

func TestOptimizer_1(t *testing.T) {
	softmax_outputs := mat.NewDense(3, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08})
	class_targets := mat.NewDense(1, 3, []float64{0, 1, 1})

	softmax_loss := new(SoftmaxCatCrossEntropy)
	softmax_loss.Backward(softmax_outputs, class_targets)
	d_values_1 := softmax_loss.D_Inputs

	activation_1 := new(SoftMax)
	activation_1.Output = softmax_outputs
	loss_value := new(loss.CategoricalCrossEntropy)
	loss_value.Backward(softmax_outputs, class_targets)
	activation_1.Backward(loss_value.D_Inputs)
	d_values_2 := activation_1.D_Inputs

	fmt.Println("Gradients: combined loss and activation")
	fmt.Println(mat.Formatted(d_values_1))
	fmt.Println()
	fmt.Println("Gradients: separate loss and activation")
	fmt.Println(mat.Formatted(d_values_2))
}
