package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/saent-x/ids-nn/core/optimization"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
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

	loss_activation := activation.CreateSoftmaxCatCrossEntropy()

	layer_1.Forward(X)
	activation_1.Forward(layer_1.Output)

	layer_2.Forward(activation_1.Output)

	loss_value := loss_activation.Forward(layer_2.Output, y)
	accuracy_ := accuracy.Calculate(loss_activation.Output, y)

	fmt.Println("loss: ", loss_value)
	fmt.Println("accuracy: ", accuracy_)

	assert.NotNil(t, accuracy_)
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

func TestEpoch_1(t *testing.T) {
	//X, y := core.SpiralData(100, 3)
	X, y := core.MockTestData()

	//X := mat.NewDense(3, 2, []float64{0.7, 0.2, 0.5, 0.1, 0.02, 0.9})
	//y := mat.NewDense(1, 3, []float64{0, 1, 1})

	layer_1 := layer.MockLayer64(2, 64)
	layer_2 := layer.MockLayer64(64, 3)

	activation_1 := new(activation.ReLU)

	loss_activation := activation.CreateSoftmaxCatCrossEntropy()
	//optimizer := optimization.CreateStochasticGradientDescent(1., .001, .9)
	//optimizer := optimization.CreateAdaptiveGradient(1., .0001, .0000001)
	optimizer := optimization.CreateRootMeanSquarePropagation(0.02, 0.00001, 0.0000001, 0.999)

	for epoch, _ := range lo.Range(10001) {
		layer_1.Forward(X)

		activation_1.Forward(layer_1.Output)
		layer_2.Forward(activation_1.Output)

		loss_value := loss_activation.Forward(layer_2.Output, y)
		accuracy_ := accuracy.Calculate(loss_activation.Output, y)

		if epoch%100 == 0 {
			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f, lr: %f\n", epoch, accuracy_, loss_value, optimizer.CurrentLearningRate)
		}

		loss_activation.Backward(loss_activation.Output, y)

		//fmt.Println("loss activation")
		//fmt.Println(mat.Formatted(core.FirstN(loss_activation.D_Inputs, 5)))
		//fmt.Println("")
		layer_2.Backward(loss_activation.D_Inputs)
		//fmt.Println("layer 2 D_Inputs")
		//fmt.Println(mat.Formatted(core.FirstN(layer_2.D_Inputs, 5)))
		//fmt.Println("")

		activation_1.Backward(layer_2.D_Inputs)
		//fmt.Println("activation 1 D_Inputs")
		//fmt.Println(mat.Formatted(core.FirstN(activation_1.D_Inputs, 5)))
		//fmt.Println("")
		layer_1.Backward(activation_1.D_Inputs)
		//fmt.Println("Layer 1 D_Inputs")
		//fmt.Println(mat.Formatted(core.FirstN(layer_1.D_Inputs, 5)))
		//fmt.Println(" ")

		optimizer.PreUpdateParams()
		optimizer.UpdateParams(layer_1)
		optimizer.UpdateParams(layer_2)
		optimizer.PostUpdateParams()

		//fmt.Println("Layer 1 D_Weights")
		//fmt.Println(mat.Formatted(core.FirstN(layer_1.D_Weights, 2)))
		//fmt.Println(" ")
		//fmt.Println("Layer 1 D_Biases")
		//fmt.Println(mat.Formatted(layer_1.D_Biases))
		//fmt.Println(" ")
		//
		//fmt.Println("Layer 2 D_Weights")
		//fmt.Println(mat.Formatted(layer_2.D_Weights))
		//fmt.Println(" ")
		//fmt.Println("Layer 2 D_Biases")
		//fmt.Println(mat.Formatted(layer_2.D_Biases))

		//fmt.Printf("-----------------------------------------------------------------\n\n")
	}

	//assert.NotNil(t, accuracy_)
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

func TestLayerCreation(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08, 0.02, 0.9, 0.08})

	layer_1 := layer.CreateLayer(3, 3)
	layer_2 := layer.CreateLayer(3, 3)

	activation_1 := new(activation.ReLU)
	opt := optimization.CreateStochasticGradientDescent(1.0, 0.02, .5)

	layer_1.Forward(X)
	fmt.Println(mat.Formatted(layer_1.Output))
	fmt.Println(" ")

	activation_1.Forward(layer_1.Output)
	fmt.Println("activation output: ", mat.Formatted(activation_1.Output))

	activation_1.Backward(activation_1.Output)
	fmt.Println("activation dinputs: ", mat.Formatted(activation_1.D_Inputs))
	layer_1.Backward(activation_1.D_Inputs)

	fmt.Println("layer 1 dinputs: ", mat.Formatted(layer_1.D_Inputs))

	opt.UpdateParams(layer_1)
	fmt.Println(" ")
	fmt.Println("layer 1 weights: ", mat.Formatted(layer_1.Weights))
	fmt.Println("layer 1 biases: ", mat.Formatted(layer_1.Biases))

	if layer_1 == nil || layer_2 == nil {
		t.Errorf("error: layer_1 & layer_2 are nil!")
	}
}
