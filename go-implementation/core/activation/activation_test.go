package activation

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/layer"
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

	layer_1 := layer.CreateLayer(2, 3)
	layer_2 := layer.CreateLayer(3, 3)

	activation_1 := new(ReLU)
	activation_2 := new(SoftMax)

	layer_1.Forward(X)
	activation_1.Forward(layer_1.Output)
	layer_2.Forward(activation_1.Output)
	activation_2.Forward(layer_2.Output)

	if activation_2.Output == nil {
		t.Errorf("error: activation_2 is nil and invalid!")
	}
}

func TestSoftmaxFunction(t *testing.T) {
	inputs := mat.NewDense(3, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08})

	softmax := new(SoftMax)
	softmax.Forward(inputs)

	fmt.Println(mat.Formatted(softmax.Output))

	if softmax.Output == nil {
		t.Errorf("error: softmax output is nil!")
	}
}
