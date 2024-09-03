package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"testing"
)

func TestNeuralNetworkLossFunction_1(t *testing.T) {
	X, y := core.SpiralData(100, 3)

	layer_1 := layer.CreateLayer(2, 3)
	layer_2 := layer.CreateLayer(3, 3)

	activation_1 := new(activation.ReLU)
	activation_2 := new(activation.SoftMax) // for the output layer

	lossfn_1 := new(loss.CrossEntropyLossFunction)

	layer_1.Forward(X)
	activation_1.Forward(layer_1.Output)

	layer_2.Forward(activation_1.Output)
	activation_2.Forward(layer_2.Output)

	r, c := activation_2.Output.Dims()
	fmt.Println(mat.Formatted(activation_2.Output.Slice(r-5, r, 0, c)))

	loss_value := lossfn_1.Calculate(activation_2.Output, y)

	fmt.Println("loss: ", loss_value)
}

func TestMisc(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	fmt.Println(stat.Mean(a, nil))
}
