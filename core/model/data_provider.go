package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	datawrappers "github.com/saent-x/ids-nn/core/model/data_wrappers"
	"github.com/saent-x/ids-nn/core/optimization"
	"gonum.org/v1/gonum/mat"
	"io"
	"os"
	"reflect"
)

type ModelDataProvider struct {
}

func (modelDataProvider *ModelDataProvider) Save(filename string, model *Model) error {
	modelFile, err := os.Create(fmt.Sprintf("./saved_models/%v.json", filename))
	if err != nil {
		return err
	}
	defer modelFile.Close()

	layers := make([]datawrappers.LayerWrapper, 0)
	for i := 0; i < len(model.Layers); i++ {
		modelLayer := model.Layers[i]
		if l, ok := modelLayer.(*layer.Layer); ok {
			lw := datawrappers.LayerWrapper{
				Type: reflect.TypeOf(modelLayer).String(),
				Weights: datawrappers.MatDenseWrapper{
					Data: l.Weights.RawMatrix().Data,
					Rows: l.Weights.RawMatrix().Rows,
					Cols: l.Weights.RawMatrix().Cols,
				},
				Biases: datawrappers.MatDenseWrapper{
					Data: l.Biases.RawMatrix().Data,
					Rows: l.Biases.RawMatrix().Rows,
					Cols: l.Biases.RawMatrix().Cols,
				},
				Weight_Regularizer_L1: l.Weight_Regularizer_L1,
				Weight_Regularizer_L2: l.Weight_Regularizer_L2,
				Biases_Regularizer_L1: l.Biases_Regularizer_L1,
				Biases_Regularizer_L2: l.Biases_Regularizer_L2,
			}

			layers = append(layers, lw)
		}
		if l, ok := model.Layers[i].(*activation.ReLU); ok {
			lw := datawrappers.LayerWrapper{
				Type: reflect.TypeOf(l).String(),
			}
			layers = append(layers, lw)
		}
		if l, ok := model.Layers[i].(*activation.Linear); ok {
			lw := datawrappers.LayerWrapper{
				Type: reflect.TypeOf(l).String(),
			}
			layers = append(layers, lw)
		}
		if l, ok := model.Layers[i].(*activation.Sigmoid); ok {
			lw := datawrappers.LayerWrapper{
				Type: reflect.TypeOf(l).String(),
			}
			layers = append(layers, lw)
		}
		if l, ok := model.Layers[i].(*activation.SoftMax); ok {
			lw := datawrappers.LayerWrapper{
				Type: reflect.TypeOf(l).String(),
			}
			layers = append(layers, lw)
		}
		if l, ok := model.Layers[i].(*activation.SoftmaxCatCrossEntropy); ok {
			lw := datawrappers.LayerWrapper{
				Type: reflect.TypeOf(l).String(),
			}
			layers = append(layers, lw)
		}
	}

	optimizer := datawrappers.OptimizerWrapper{
		Type: reflect.TypeOf(model.Optimizer).String(),
		Obj:  model.Optimizer,
	}

	modelWrapper := datawrappers.ModelWrapper{
		layers,
		reflect.TypeOf(model.Lossfn).String(),
		reflect.TypeOf(model.Accuracy).String(),
		optimizer,
	}

	d, err := json.MarshalIndent(modelWrapper, "", "  ")

	_, err = modelFile.Write(d)
	if err != nil {
		return err
	}

	return nil
}

func (modelDataProvider *ModelDataProvider) Load(filename string) (*Model, error) {
	modelFile, err := os.Open(fmt.Sprintf("./saved_models/%v.json", filename))
	if err != nil {
		panic(err)
	}
	defer modelFile.Close()

	bytesData, err := io.ReadAll(modelFile)
	if err != nil {
		panic(err)
	}

	var retrievedModel datawrappers.ModelWrapper
	err = json.Unmarshal(bytesData, &retrievedModel)
	if err != nil {
		panic(err)
	}

	model := Model{}
	// fill model layers

	for i := 0; i < len(retrievedModel.Layers); i++ {
		layer_ := retrievedModel.Layers[i]

		if layer_.Type == reflect.TypeOf(&layer.Layer{}).String() {
			w := layer_.Weights
			b := layer_.Biases

			l := layer.Layer{
				Weights: mat.NewDense(w.Rows, w.Cols, w.Data),
				Biases:  mat.NewDense(b.Rows, b.Cols, b.Data),

				Weight_Regularizer_L1: layer_.Weight_Regularizer_L1,
				Weight_Regularizer_L2: layer_.Weight_Regularizer_L2,
				Biases_Regularizer_L1: layer_.Biases_Regularizer_L1,
				Biases_Regularizer_L2: layer_.Biases_Regularizer_L2,
			}
			model.Add(&l)
		}
		if layer_.Type == reflect.TypeOf(&activation.ReLU{}).String() {
			model.Add(&activation.ReLU{})
		}
		if layer_.Type == reflect.TypeOf(&activation.Linear{}).String() {
			model.Add(&activation.Linear{})
		}
		if layer_.Type == reflect.TypeOf(&activation.Sigmoid{}).String() {
			model.Add(&activation.Sigmoid{})
		}
		if layer_.Type == reflect.TypeOf(&activation.SoftMax{}).String() {
			model.Add(&activation.SoftMax{})
		}
		if layer_.Type == reflect.TypeOf(&activation.SoftmaxCatCrossEntropy{}).String() {
			//model.Add(activation.SoftmaxCatCrossEntropy{})
			panic("un-implemented func/mapping")
		}

	}

	var lossfn loss.ILoss
	switch retrievedModel.Loss {
	case reflect.TypeOf(&loss.CategoricalCrossEntropy{}).String():
		lossfn = &loss.CategoricalCrossEntropy{}
	case reflect.TypeOf(&loss.BinaryCrossEntropy{}).String():
		lossfn = &loss.BinaryCrossEntropy{}
	case reflect.TypeOf(&loss.MeanSquaredError{}).String():
		lossfn = &loss.MeanSquaredError{}
	case reflect.TypeOf(&loss.MeanAbsoluteError{}).String():
		lossfn = &loss.MeanAbsoluteError{}
	default:
		return (&Model{}), errors.New("invalid loss value")
	}

	var accuracy_ accuracy.IAccuracy
	switch retrievedModel.Accuracy {
	case reflect.TypeOf(&accuracy.CategoricalAccuracy{}).String():
		accuracy_ = &accuracy.CategoricalAccuracy{}
	case reflect.TypeOf(&accuracy.BinaryAccuracy{}).String():
		accuracy_ = &accuracy.BinaryAccuracy{}
	case reflect.TypeOf(&accuracy.RegressionAccuracy{}).String():
		accuracy_ = &accuracy.RegressionAccuracy{}
	default:
		return (&Model{}), errors.New("invalid accuracy value")
	}

	var optimizer optimization.IOptimizer
	switch retrievedModel.Optimizer.Type {
	case reflect.TypeOf(&optimization.AdaptiveMomentum{}).String():
		optData, err := json.Marshal(retrievedModel.Optimizer.Obj)

		if err != nil {
			return (&Model{}), err
		}
		var result optimization.AdaptiveMomentum

		err = json.Unmarshal(optData, &result)
		if err != nil {
			return (&Model{}), err
		}

		optimizer = &result
	case reflect.TypeOf(&optimization.AdaptiveGradient{}).String():
		optData, err := json.Marshal(retrievedModel.Optimizer.Obj)

		if err != nil {
			return (&Model{}), err
		}
		var result optimization.AdaptiveGradient

		err = json.Unmarshal(optData, &result)
		if err != nil {
			return (&Model{}), err
		}

		optimizer = &result
	case reflect.TypeOf(&optimization.StochasticGradientDescent{}).String():
		optData, err := json.Marshal(retrievedModel.Optimizer.Obj)

		if err != nil {
			return (&Model{}), err
		}
		var result optimization.StochasticGradientDescent

		err = json.Unmarshal(optData, &result)
		if err != nil {
			return (&Model{}), err
		}

		optimizer = &result
	case reflect.TypeOf(&optimization.RootMeanSquarePropagation{}).String():
		optData, err := json.Marshal(retrievedModel.Optimizer.Obj)

		if err != nil {
			return (&Model{}), err
		}
		var result optimization.RootMeanSquarePropagation

		err = json.Unmarshal(optData, &result)
		if err != nil {
			return (&Model{}), err
		}

		optimizer = &result
	default:
		return (&Model{}), errors.New("invalid accuracy value")
	}

	model.Set(lossfn, optimizer, accuracy_)
	model.Finalize()

	return &model, nil
}
