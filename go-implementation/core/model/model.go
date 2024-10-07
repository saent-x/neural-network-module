package model

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/datamodels"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/saent-x/ids-nn/core/optimization"
	"github.com/saent-x/ids-nn/core/serializer"
	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Layers                  []any
	TrainableLayers         []*layer.Layer
	Lossfn                  loss.ILoss
	Optimizer               optimization.IOptimizer
	InputLayer              *layer.InputLayer
	OutputLayerActivation   activation.IActivation
	Accuracy                accuracy.IAccuracy
	SoftMaxClassifierOutput *activation.SoftmaxCatCrossEntropy
}

func New() *Model {
	return &Model{
		Layers:                  []any{},
		SoftMaxClassifierOutput: nil,
		Lossfn:                  nil,
	}
}

func (model *Model) Add(layer layer.ILayer) {
	model.Layers = append(model.Layers, layer)
}

func (model *Model) Set(lossfn loss.ILoss, optimizer optimization.IOptimizer, accuracy accuracy.IAccuracy) {
	if lossfn != nil {
		model.Lossfn = lossfn
	}
	if optimizer != nil {
		model.Optimizer = optimizer
	}
	if accuracy != nil {
		model.Accuracy = accuracy
	}
}

func (model *Model) Train(training_data datamodels.TrainingData, validation_data datamodels.ValidationData, epochs int, batch_size int, print_every int) {
	model.Accuracy.Init(training_data.Y, false)

	var train_steps int
	train_steps = 1

	if batch_size > 0 {
		len_X := training_data.X.RawMatrix().Rows
		train_steps = len_X / batch_size

		if train_steps*batch_size < len_X {
			train_steps += 1
		}
	}
	for epoch := 1; epoch < epochs+1; epoch++ {
		fmt.Println("epoch: ", epoch)

		// reset accumulated loss and accuracy
		model.Lossfn.NewPass()
		model.Accuracy.NewPass()

		for _, step := range core.GetRange(train_steps) {
			var batch_X, batch_Y *mat.Dense
			if batch_size <= 0 {
				batch_X = training_data.X
				batch_Y = training_data.Y
			} else {
				batch_X, batch_Y = core.GetBatch(training_data, step, batch_size)
			}

			output := model.forward(batch_X, true)

			data_loss, regularization_loss := model.Lossfn.Calculate(output, batch_Y, true)
			loss_value := data_loss + regularization_loss

			predictions := model.OutputLayerActivation.Predictions(output)
			accuracy_ := model.Accuracy.Calculate(predictions, batch_Y)

			model.Backward(output, batch_Y)

			model.Optimizer.PreUpdateParams()
			for i := 0; i < len(model.TrainableLayers); i++ {
				model.Optimizer.UpdateParams(model.TrainableLayers[i])
			}
			model.Optimizer.PostUpdateParams()

			if step%print_every == 0 || step == train_steps-1 {
				fmt.Printf("step: %d, acc: %.3f, loss: %.3f, data-loss: %.3f, rg-loss: %.3f, lr: %f\n", step, accuracy_, loss_value, data_loss, regularization_loss, model.Optimizer.GetCurrentLearningRate())
			}
		}

		epoch_data_loss, epoch_regularization_loss := model.Lossfn.CalculateAccumulated(true)
		epoch_loss := epoch_data_loss + epoch_regularization_loss
		epoch_accuracy := model.Accuracy.CalculateAccumulated()

		fmt.Printf("training -> acc: %.3f, loss: %.3f, data-loss: %.3f, rg-loss: %.3f, lr: %f\n", epoch_accuracy, epoch_loss, epoch_data_loss, epoch_regularization_loss, model.Optimizer.GetCurrentLearningRate())

		if validation_data != (datamodels.ValidationData{}) {
			model.Evaluate(validation_data, batch_size)
		}
	}
}

func (model *Model) forward(X *mat.Dense, training bool) *mat.Dense {
	model.InputLayer.Forward(X, training)

	var output *mat.Dense
	for i := 0; i < len(model.Layers); i++ {
		// since only the prev layer is called it shouldn't affect the loss func as the next layer to the last layer
		model.Layers[i].(layer.ILayer).Forward(model.Layers[i].(layer.ILayer).GetPreviousLayer().(layer.ILayer).GetOutput(), training)
		output = model.Layers[i].(layer.ILayer).GetOutput()
	}

	return output
}

func (model *Model) Backward(output, y *mat.Dense) {
	if model.SoftMaxClassifierOutput != nil {
		model.SoftMaxClassifierOutput.Backward(output, y)
		model.Layers[len(model.Layers)-1].(layer.ILayer).SetDInputs(model.SoftMaxClassifierOutput.GetDInputs())

		for i := len(model.Layers) - 1; i >= 0; i-- {
			// lets skip the first value (which is same as last in the model.Layers slice)
			if i != len(model.Layers)-1 {
				model.Layers[i].(layer.ILayer).Backward(model.Layers[i].(layer.ILayer).GetNextLayer().(layer.ILayer).GetDInputs())
			}
		}
		return
	}

	model.Lossfn.Backward(output, y)

	for i := len(model.Layers) - 1; i >= 0; i-- {
		if i == len(model.Layers)-1 {
			model.Layers[i].(layer.ILayer).Backward(model.Lossfn.GetDInputs())
		} else {
			model.Layers[i].(layer.ILayer).Backward(model.Layers[i].(layer.ILayer).GetNextLayer().(layer.ILayer).GetDInputs())
		}
	}
}

func (model *Model) Finalize() {
	model.InputLayer = new(layer.InputLayer)
	layers_count := len(model.Layers)

	model.TrainableLayers = []*layer.Layer{}

	for i := 0; i < layers_count; i++ {
		// first layer
		if i == 0 {
			if layer, ok := model.Layers[i].(layer.ILayer); ok {
				layer.SetPreviousLayer(model.InputLayer)
				layer.SetNextLayer(model.Layers[i+1])
				// if it doesn't work try overwriting the layer
			}
		} else if i < layers_count-1 /** all layers asides 1st and last**/ {
			if layer, ok := model.Layers[i].(layer.ILayer); ok {
				layer.SetPreviousLayer(model.Layers[i-1])
				layer.SetNextLayer(model.Layers[i+1])
			}
		} else {
			if layer, ok := model.Layers[i].(layer.ILayer); ok {
				layer.SetPreviousLayer(model.Layers[i-1])
				layer.SetNextLayer(model.Lossfn)

				model.OutputLayerActivation = layer.(activation.IActivation)
			}
		}

		// since only layers have Weights
		if layer, ok := model.Layers[i].(*layer.Layer); ok {
			model.TrainableLayers = append(model.TrainableLayers, layer)
		}
	}
	if model.Lossfn != nil {
		model.Lossfn.RememberTrainableLayers(model.TrainableLayers)
	}

	_, isSoftmax := model.Layers[len(model.Layers)-1].(activation.SoftMax)
	_, isCatCrossEntropy := model.Lossfn.(*loss.CategoricalCrossEntropy)

	if isSoftmax && isCatCrossEntropy {
		model.SoftMaxClassifierOutput = new(activation.SoftmaxCatCrossEntropy)
	}

}

func (model *Model) Evaluate(validation_data datamodels.ValidationData, batch_size int) {
	validation_steps := 1

	if batch_size > 0 {
		len_X_val := validation_data.X.RawMatrix().Rows
		validation_steps = len_X_val / batch_size

		if validation_steps*batch_size < len_X_val {
			validation_steps += 1
		}
	}

	model.Lossfn.NewPass()
	model.Accuracy.NewPass()

	for _, step := range core.GetRange(validation_steps) {
		var batch_X_val, batch_Y_val *mat.Dense
		if batch_size <= 0 {
			batch_X_val = validation_data.X
			batch_Y_val = validation_data.Y
		} else {
			batch_X_val, batch_Y_val = core.GetBatch(validation_data, step, batch_size)
		}
		output := model.forward(batch_X_val, false)

		_, _ = model.Lossfn.Calculate(output, batch_Y_val, false)
		predictions := model.OutputLayerActivation.Predictions(output)
		_ = model.Accuracy.Calculate(predictions, batch_Y_val)
	}

	validation_loss, _ := model.Lossfn.CalculateAccumulated(false)
	validation_accuracy := model.Accuracy.CalculateAccumulated()

	fmt.Printf("\nValidation -> acc: %f loss: %f\n\n", validation_accuracy, validation_loss)
}

func (model *Model) getParameters() []datamodels.ModelParameter {
	modelParameters := []datamodels.ModelParameter{}

	for _, layer := range model.TrainableLayers {
		modelParameters = append(modelParameters, layer.GetParameters())
	}

	return modelParameters
}

func (model *Model) SetParameters(modelParameters []datamodels.ModelParameter) {
	for i := 0; i < len(model.TrainableLayers); i++ {
		model.TrainableLayers[i].SetParameters(modelParameters[i])
	}
}

func (model *Model) SaveParameters(filename string) {
	err := serializer.Serialize(filename, model.getParameters())
	if err != nil {
		fmt.Println("error serializing model", err)
	}
}

func (model *Model) LoadParameters(filename string) {
	var data []datamodels.ModelParameter

	err := serializer.Deserialize(filename, &data)
	if err != nil {
		fmt.Println("error deserializing model", err)
	}

	model.SetParameters(data)
}

func (model *Model) Predict(X *mat.Dense, batchSize int) *mat.Dense {
	predictionSteps := 1

	if batchSize > 0 {
		lenX := X.RawMatrix().Rows
		predictionSteps = lenX / batchSize

		if predictionSteps*batchSize < lenX {
			predictionSteps += 1
		}
	}

	var outputs [][]float64
	for _, step := range core.GetRange(predictionSteps) {
		var batch_X *mat.Dense
		if batchSize <= 0 {
			batch_X = X
		} else {
			batch_X = core.GetSingleBatch(X, step, batchSize)
		}
		batchOutput := model.forward(batch_X, false)
		for x := 0; x < batchOutput.RawMatrix().Rows; x++ {
			row := mat.Row(nil, x, batchOutput)
			outputs = append(outputs, row)
		}
	}

	matrix := mat.NewDense(len(outputs), len(outputs[0]), nil)

	// convert outputs to *mat.Dense
	for i := 0; i < len(outputs); i++ {
		matrix.SetRow(i, outputs[i])
	}

	return matrix
}
