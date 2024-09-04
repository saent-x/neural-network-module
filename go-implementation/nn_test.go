package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
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

	loss_value := lossfn_1.Calculate(activation_2.Output, y)

	fmt.Println("loss: ", loss_value)

	if loss_value != 0.34 {
		t.Errorf("error: got %f | want %f", loss_value, 1.09861)
	}
}

func TestNeuralNetworkAccuracyFunction_1(t *testing.T) {
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

	loss_value := lossfn_1.Calculate(activation_2.Output, y)
	accuracy_ := accuracy.Calculate(activation_2.Output, y)

	fmt.Println("loss: ", loss_value)
	fmt.Println("accuracy: ", accuracy_)

	if accuracy_ != 0.34 {
		t.Errorf("error: got %f | want %f", accuracy_, 0.34)
	}
}

func TestNeuralNetworkLossFunction_2(t *testing.T) {
	softmax_output := mat.NewDense(3, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08})
	class_target := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 1, 0})

	cross_entropy := new(loss.CrossEntropyLossFunction)
	loss_value := cross_entropy.Calculate(softmax_output, class_target)

	got := loss_value
	want := 0.38506088005216804

	if got != want {
		t.Errorf("error: got %f | want %f", got, want)
	}

	fmt.Println(got)
}

func TestMisc(t *testing.T) {
	X, y := core.SpiralData(100, 3)

	r, _ := X.Dims()

	lo.ForEach(lo.Range(r), func(item int, index int) {
		row := X.RawRowView(index)
		fmt.Printf("[%f, %f], ", row[0], row[1])
	})

	_, c := y.Dims()

	fmt.Println()

	lo.ForEach(lo.Range(c), func(item int, index int) {
		fmt.Printf("%d, ", int(y.At(0, index)))
	})

}
