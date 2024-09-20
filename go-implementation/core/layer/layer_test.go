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

func TestSimple(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	b := mat.NewDense(2, 2, []float64{4, 3, 2, 1})

	fmt.Println(mat.Formatted(a))
	fmt.Println(mat.Formatted(b))

	fmt.Printf("\nafter\n")
	b = mat.DenseCopyOf(a)

	fmt.Println(mat.Formatted(a))
	fmt.Println(mat.Formatted(b))
}
