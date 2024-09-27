package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/optimization"
	"github.com/samber/lo"
)

func main() {
	NNTraining_1()
}

func NNTraining_1() {
	X, y := core.MockTestData2()

	layer_1 := layer.MockLayer64_1000(2, 64, 0, 0.0005, 0, 0.0005)
	layer_2 := layer.MockLayer64_1000(64, 3, 0, 0, 0, 0)
	dropout_layer := layer.NewDropoutLayer(0.1)
	activation_1 := new(activation.ReLU)

	loss_activation := activation.CreateSoftmaxCatCrossEntropy()
	optimizer := optimization.CreateAdaptiveMomentum(0.05, 0.00005, 0.0000001, 0.9, 0.999)

	for epoch, _ := range lo.Range(10001) {
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

	nnValidation(layer_1, layer_2, activation_1, loss_activation)
}

func nnValidation(layer_1 *layer.Layer, layer_2 *layer.Layer, activation_1 *activation.ReLU, loss_activation *activation.SoftmaxCatCrossEntropy) {
	X_test, y_test := core.SpiralData(100, 3)

	layer_1.Forward(X_test)
	activation_1.Forward(layer_1.Output)
	layer_2.Forward(activation_1.Output)

	loss_value := loss_activation.Calculate(layer_2.Output, y_test)
	accuracy_ := accuracy.Calculate(loss_activation.Output, y_test)

	fmt.Printf("\nvalidation, acc: %.3f, loss: %.3f", accuracy_, loss_value)
}
