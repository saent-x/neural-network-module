package activation

import (
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/layer"
	"testing"
)

func TestActivationCreation(t *testing.T) {
	activation_1 := new(ReLU)
	activation_2 := new(SoftMax)

	if activation_1 == nil || activation_2 == nil {
		t.Errorf("error: activation_1 & activation_2 are nil!")
	}
}

func TestForwardFunction(t *testing.T) {
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

	// r, c := activation_2.Output.Dims()
	// last_5_rows := activation_2.Output.Slice(r-5, r, 0, c)
	// fmt.Println(mat.Formatted(last_5_rows))
}
