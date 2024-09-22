package layer

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestReLUBackwardFunction(t *testing.T) {
	inputs := mat.NewDense(3, 2, []float64{
		1.0, 2.0,
		0.5, -0.5,
		-1.0, 2.0,
	})
	weights := mat.NewDense(2, 3, []float64{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
	})
	biases := mat.NewDense(1, 3, []float64{0.0, 0.0, 0.0})
	dvalues := mat.NewDense(3, 3, []float64{
		0.1, -0.2, 0.3,
		0.4, -0.5, 0.6,
		-0.7, 0.8, -0.9,
	})

	layer := Layer{
		Inputs:  inputs,
		Weights: weights,
		Biases:  biases,
	}

	layer.Backward(dvalues)

	fmt.Printf("DWeights:\n%v\n", mat.Formatted(layer.D_Weights, mat.Prefix(" "), mat.Squeeze()))
	fmt.Printf("DBiases:\n%v\n", mat.Formatted(layer.D_Biases, mat.Prefix(" "), mat.Squeeze()))
	fmt.Printf("DInputs:\n%v\n", mat.Formatted(layer.D_Inputs, mat.Prefix(" "), mat.Squeeze()))
}

func TestMisc(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08, 0.02, 0.9, 0.08})

	layer_1 := NewDropoutLayer(0.1)
	layer_1.Forward(X)

	fmt.Println(mat.Formatted(layer_1.Output))
}

func TestLayerCreation(t *testing.T) {
	layer_1 := CreateLayer(3, 3, 0, 0, 0, 0)
	layer_2 := CreateLayer(3, 3, 0, 0, 0, 0)

	if layer_1 == nil || layer_2 == nil {
		t.Errorf("error: layer_1 & layer_2 are nil!")
	}
}
