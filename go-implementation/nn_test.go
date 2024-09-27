package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/saent-x/ids-nn/core/optimization"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"testing"
)

func TestNeuralNetworkLossFunction_1(t *testing.T) {
	X, y := core.SpiralData(100, 3)

	layer_1 := layer.CreateLayer(2, 3, 0, 0, 0, 0)
	layer_2 := layer.CreateLayer(3, 3, 0, 0, 0, 0)

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

	layer_1 := layer.CreateLayer(2, 3, 0, 0, 0, 0)
	layer_2 := layer.CreateLayer(3, 3, 0, 0, 0, 0)

	activation_1 := new(activation.ReLU)

	loss_activation := activation.CreateSoftmaxCatCrossEntropy()

	layer_1.Forward(X)
	activation_1.Forward(layer_1.Output)

	layer_2.Forward(activation_1.Output)

	loss_value := loss_activation.Calculate(layer_2.Output, y)
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

// Classfication - OHE
func TestNNTraining_1(t *testing.T) {
	X, y := core.MockTestData2()

	layer_1 := layer.MockLayer64_1000(2, 64, 0, 0.0005, 0, 0.0005)
	layer_2 := layer.MockLayer64_1000(64, 3, 0, 0, 0, 0)
	dropout_layer := layer.NewDropoutLayer(0.1)
	activation_1 := new(activation.ReLU)

	loss_activation := activation.CreateSoftmaxCatCrossEntropy()
	optimizer := optimization.CreateAdaptiveMomentum(0.05, 0.00005, 0.0000001, 0.9, 0.999)

	for epoch := 0; epoch < 10001; epoch++ {
		layer_1.Forward(X)

		activation_1.Forward(layer_1.Output)
		dropout_layer.Forward(activation_1.Output)
		layer_2.Forward(dropout_layer.Output)

		data_loss := loss_activation.Calculate(layer_2.Output, y)
		regularization_loss := loss_activation.CalcRegularizationLoss_1(layer_1) + loss_activation.CalcRegularizationLoss_1(layer_2)
		loss_value := data_loss + regularization_loss

		accuracy_ := accuracy.Calculate(loss_activation.Output, y)

		if epoch%100 == 0 {
			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f, data-loss: %.3f, rg-loss: %.3f, lr: %f\n", epoch, accuracy_, loss_value, data_loss, regularization_loss, optimizer.CurrentLearningRate)
		}

		loss_activation.Backward(loss_activation.Output, y)
		layer_2.Backward(loss_activation.D_Inputs)
		dropout_layer.Backward(layer_2.D_Inputs)
		activation_1.Backward(dropout_layer.D_Inputs)
		layer_1.Backward(activation_1.D_Inputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParams(layer_1)
		optimizer.UpdateParams(layer_2)
		optimizer.PostUpdateParams()
	}

	NNValidation(layer_1, layer_2, activation_1, loss_activation)
}

func NNValidation(layer_1 *layer.Layer, layer_2 *layer.Layer, activation_1 *activation.ReLU, loss_activation *activation.SoftmaxCatCrossEntropy) {
	X_test, y_test := core.SpiralData(100, 3)

	layer_1.Forward(X_test)
	activation_1.Forward(layer_1.Output)
	layer_2.Forward(activation_1.Output)

	loss_value := loss_activation.Calculate(layer_2.Output, y_test)
	accuracy_ := accuracy.Calculate(loss_activation.Output, y_test)

	fmt.Printf("\nvalidation, acc: %.3f, loss: %.3f", accuracy_, loss_value)
}

func NNValidation_2(layer_1 *layer.Layer, layer_2 *layer.Layer, activation_1 *activation.ReLU, activation_2 *activation.Sigmoid, loss_fn *loss.BinaryCrossEntropyLoss) {
	X_test, y_test := core.SpiralData(100, 2)

	y_test_reshape := mat.NewDense(y_test.RawMatrix().Cols, y_test.RawMatrix().Rows, y_test.RawMatrix().Data)

	layer_1.Forward(X_test)
	activation_1.Forward(layer_1.Output)
	layer_2.Forward(activation_1.Output)
	activation_2.Forward(layer_2.Output)

	loss_value := loss_fn.Calculate(activation_2.Output, y_test_reshape)
	accuracy_ := accuracy.BinaryCalculate(layer_2.Output, y_test_reshape, 0.5)

	fmt.Printf("\nvalidation, acc: %.3f, loss: %.3f", accuracy_, loss_value)
}

// Binary Logistic Regression
func TestNNTraining_2(t *testing.T) {
	X, y := core.BinaryMockTestData()

	layer_1 := layer.MockLayer64(2, 64, 0, 0.0005, 0, 0.0005)
	layer_2 := layer.MockLayer64(64, 1, 0, 0, 0, 0)

	activation_1 := new(activation.ReLU)
	activation_2 := new(activation.Sigmoid)

	loss_function := new(loss.BinaryCrossEntropyLoss)
	optimizer := optimization.CreateAdaptiveMomentum(0.01, 0.0000005, 0.0000001, 0.9, 0.999)

	for epoch := 0; epoch < 10001; epoch++ {
		layer_1.Forward(X)

		activation_1.Forward(layer_1.Output)
		layer_2.Forward(activation_1.Output)
		activation_2.Forward(layer_2.Output)

		data_loss := loss_function.Calculate(activation_2.Output, y)
		regularization_loss := loss_function.CalcRegularizationLoss_1(layer_1) + loss_function.CalcRegularizationLoss_1(layer_2)
		loss_value := data_loss + regularization_loss

		accuracy_ := accuracy.BinaryCalculate(activation_2.Output, y, 0.5)

		if epoch%100 == 0 {
			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f, data-loss: %.3f, rg-loss: %.3f, lr: %f\n", epoch, accuracy_, loss_value, data_loss, regularization_loss, optimizer.CurrentLearningRate)
		}

		loss_function.Backward(activation_2.Output, y)
		activation_2.Backward(loss_function.D_Inputs)
		layer_2.Backward(activation_2.D_Inputs)
		activation_1.Backward(layer_2.D_Inputs)
		layer_1.Backward(activation_1.D_Inputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParams(layer_1)
		optimizer.UpdateParams(layer_2)
		optimizer.PostUpdateParams()
	}

	NNValidation_2(layer_1, layer_2, activation_1, activation_2, loss_function)
}

func TestForRegressionModel(t *testing.T) {
	X, y := core.SineData(1000)

	layer_1 := layer.MockRegressionLayer(1, 64, 0, 0, 0, 0)
	activation_1 := new(activation.ReLU)
	layer_2 := layer.MockRegressionLayer(64, 64, 0, 0, 0, 0)
	activation_2 := new(activation.ReLU)
	layer_3 := layer.MockRegressionLayer(64, 1, 0, 0, 0, 0)
	activation_3 := new(activation.Linear)

	loss_function := new(loss.MeanSquaredError)
	optimizer := optimization.CreateAdaptiveMomentum(0.005, .001, 0.0000001, 0.9, 0.999)

	accuracy_precision := stat.StdDev(y.RawMatrix().Data, nil) / 250

	for epoch := 0; epoch < 10001; epoch++ {
		layer_1.Forward(X)
		activation_1.Forward(layer_1.Output)

		layer_2.Forward(activation_1.Output)
		activation_2.Forward(layer_2.Output)

		layer_3.Forward(activation_2.Output)
		activation_3.Forward(layer_3.Output)

		data_loss := loss_function.Calculate(activation_3.Output, y)

		regularization_loss := loss_function.CalcRegularizationLoss_1(layer_1) + loss_function.CalcRegularizationLoss_1(layer_2) + loss_function.CalcRegularizationLoss_1(layer_3)
		loss_value := data_loss + regularization_loss

		accuracy_ := accuracy.LinearCalculate(activation_3.Output, y, accuracy_precision)

		if epoch%100 == 0 {
			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f, data-loss: %.3f, rg-loss: %.3f, lr: %f\n", epoch, accuracy_, loss_value, data_loss, regularization_loss, optimizer.CurrentLearningRate)
		}

		loss_function.Backward(activation_3.Output, y)

		activation_3.Backward(loss_function.D_Inputs)
		layer_3.Backward(activation_3.D_Inputs)

		activation_2.Backward(layer_3.D_Inputs)
		layer_2.Backward(activation_2.D_Inputs)

		activation_1.Backward(layer_2.D_Inputs)
		layer_1.Backward(activation_1.D_Inputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParams(layer_1)
		optimizer.UpdateParams(layer_2)
		optimizer.UpdateParams(layer_3)
		optimizer.PostUpdateParams()
	}
}
