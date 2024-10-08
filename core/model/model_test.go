package model

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/datamodels"
	"github.com/saent-x/ids-nn/core/datasets"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/saent-x/ids-nn/core/mock"
	"github.com/saent-x/ids-nn/core/optimization"
	"gonum.org/v1/gonum/mat"
	"os"
	"testing"
)

func TestRegressionModel(t *testing.T) {
	X, y := core.SineData(1000)

	regression_model := New()

	regression_model.Add(mock.MockRegressionLayer(1, 64, 0, 0, 0, 0))
	regression_model.Add(new(activation.ReLU))

	regression_model.Add(mock.MockRegressionLayer(64, 64, 0, 0, 0, 0))
	regression_model.Add(new(activation.ReLU))

	regression_model.Add(mock.MockRegressionLayer(64, 1, 0, 0, 0, 0))
	regression_model.Add(new(activation.Linear))

	regression_model.Set(new(loss.MeanSquaredError), optimization.CreateAdaptiveMomentum(0.005, .001, 0.0000001, 0.9, 0.999), new(accuracy.RegressionAccuracy))

	regression_model.Finalize()

	regression_model.Train(datamodels.TrainingData{X: X, Y: y}, datamodels.ValidationData{nil, nil}, 10000, 0, 100)
}

func TestBinaryModel(t *testing.T) {
	X, y := mock.BinaryMockTestData2()
	X_test, y_test := core.SpiralData(100, 2)
	y_test_reshape := mat.NewDense(y_test.RawMatrix().Cols, y_test.RawMatrix().Rows, y_test.RawMatrix().Data)

	binary_categorical_model := New()

	binary_categorical_model.Add(mock.MockLayer64_2(2, 64, 0, 5e-4, 0, 5e-4))
	binary_categorical_model.Add(new(activation.ReLU))

	binary_categorical_model.Add(mock.MockLayer64_2(64, 1, 0, 0, 0, 0))
	binary_categorical_model.Add(new(activation.Sigmoid))

	binary_categorical_model.Set(new(loss.BinaryCrossEntropy), optimization.CreateAdaptiveMomentum(1e-3, 5e-7, 1e-7, 0.9, 0.999), new(accuracy.BinaryAccuracy))

	binary_categorical_model.Finalize()

	binary_categorical_model.Train(datamodels.TrainingData{X, y}, datamodels.ValidationData{X_test, y_test_reshape}, 10000, 0, 100)
}

func TestCategoricalModel(t *testing.T) {
	X, y := mock.MockTestData_1000()
	X_test, y_test := core.SpiralData(100, 3)

	classification_model := New()

	classification_model.Add(mock.MockLayer64_1000(2, 512, 0, 5e-4, 0, 5e-4))
	classification_model.Add(new(activation.ReLU))

	classification_model.Add(layer.NewDropoutLayer(0.1))

	classification_model.Add(mock.MockLayer64_1000(512, 3, 0, 0, 0, 0))
	classification_model.Add(new(activation.SoftMax))

	classification_model.Set(new(loss.CategoricalCrossEntropy), optimization.CreateAdaptiveMomentum(0.05, 5e-5, 1e-7, 0.9, 0.999), new(accuracy.CategoricalAccuracy))

	classification_model.Finalize()

	classification_model.Train(datamodels.TrainingData{X, y}, datamodels.ValidationData{X_test, y_test}, 10000, 0, 100)
}

func TestFashionMISTModel(t *testing.T) {
	training_data, testing_data := datasets.LoadFashionMNISTDataset(true)

	fashionMNIST_model := New()

	fashionMNIST_model.Add(layer.CreateLayer(training_data.X.RawMatrix().Cols, 128, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.ReLU))

	fashionMNIST_model.Add(layer.CreateLayer(128, 128, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.ReLU))

	fashionMNIST_model.Add(layer.CreateLayer(128, 10, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.SoftMax))

	fashionMNIST_model.Set(new(loss.CategoricalCrossEntropy), optimization.CreateAdaptiveMomentum(0.001, 1e-4, 1e-7, 0.9, 0.999), new(accuracy.CategoricalAccuracy))

	fashionMNIST_model.Finalize()

	fashionMNIST_model.Train(training_data, testing_data, 10, 128, 100)

	//fashionMNIST_model.SaveParameters("fashionMNIST_model")
	modelDataProvider := new(ModelDataProvider)
	modelDataProvider.Save("fashionMNIST_model_full", fashionMNIST_model)
}

func TestFashionMISTModelParametersFromFile(t *testing.T) {
	training_data, testing_data := datasets.LoadFashionMNISTDataset(true)

	fashionMNIST_model := New()

	fashionMNIST_model.Add(layer.CreateLayer(training_data.X.RawMatrix().Cols, 128, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.ReLU))

	fashionMNIST_model.Add(layer.CreateLayer(128, 128, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.ReLU))

	fashionMNIST_model.Add(layer.CreateLayer(128, 10, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.SoftMax))

	fashionMNIST_model.Set(new(loss.CategoricalCrossEntropy), nil, new(accuracy.CategoricalAccuracy))

	fashionMNIST_model.Finalize()
	fashionMNIST_model.LoadParameters("fashionMNIST_model")

	fashionMNIST_model.Evaluate(testing_data, 0)
}

func TestFashionMISTModelFromFile(t *testing.T) {
	_, testing_data := datasets.LoadFashionMNISTDataset(true)

	modelDataProvider := new(ModelDataProvider)

	modelFile, err := os.Open(fmt.Sprintf("./saved_models/%v.json", "fashionMNIST_model_full"))
	if err != nil {
		panic(err)
	}
	defer modelFile.Close()

	model, err := modelDataProvider.Load(modelFile)
	if err != nil {
		t.Fatal(err)
	}

	model.Evaluate(testing_data, 0)
}

func TestModelInference(t *testing.T) {
	image_data := datasets.LoadFashionMNISTDatasetForInference(false)

	modelDataProvider := new(ModelDataProvider)

	modelFile, err := os.Open(fmt.Sprintf("./saved_models/%v.json", "fashionMNIST_model_full"))
	if err != nil {
		panic(err)
	}
	defer modelFile.Close()

	model, err := modelDataProvider.Load(modelFile)
	if err != nil {
		t.Fatal(err)
	}

	confidences := model.Predict(image_data, 0)
	predictions := model.OutputLayerActivation.Predictions(confidences)

	fmt.Println(mat.Formatted(predictions))

	fashionMNIST_labels := map[int]string{
		0: "T-shirt/top",
		1: "Trouser",
		2: "Pullover",
		3: "Dress",
		4: "Coat",
		5: "Sandal",
		6: "Shirt",
		7: "Sneaker",
		8: "Bag",
		9: "Ankle boot",
	}

	for _, datum := range predictions.RawMatrix().Data {
		fmt.Println(fashionMNIST_labels[int(datum)])
	}
}

func TestModel_Predict(t *testing.T) {
	_, testing_data := datasets.LoadFashionMNISTDataset(false)

	modelDataProvider := new(ModelDataProvider)

	modelFile, err := os.Open(fmt.Sprintf("./saved_models/%v.json", "fashionMNIST_model_full"))
	if err != nil {
		panic(err)
	}
	defer modelFile.Close()

	model, err := modelDataProvider.Load(modelFile)
	if err != nil {
		t.Fatal(err)
	}

	confidences := model.Predict(core.FirstN(testing_data.X, 5), 0)
	predictions := model.OutputLayerActivation.Predictions(confidences)

	fmt.Println(mat.Formatted(predictions))
	fmt.Println()
	for i := 0; i < 5; i++ {
		fmt.Printf("%f, ", testing_data.Y.At(0, i))
	}
}

func TestSavingModelFunc(t *testing.T) {
	m := New()
	m.SaveParameters("test_model_1")
}

func TestRetrievingModelFunc(t *testing.T) {
	m := New()
	m.LoadParameters("test_model")
}

func TestCategoricalAccuracy(t *testing.T) {
	predictions := mat.NewDense(5, 1, []float64{0, 1.5, 1.5, 0, 3})
	y := mat.NewDense(5, 1, []float64{0, 0, 1.5, 0, 3})

	ca := new(accuracy.BinaryAccuracy)
	result := ca.Compare(predictions, y)

	fmt.Println(mat.Formatted(result))
}
