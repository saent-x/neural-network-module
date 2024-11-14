package model

import (
	"fmt"
	"github.com/saent-x/ids-nn/core/metrics"
	"gonum.org/v1/gonum/stat"
	"os"
	"testing"

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

	regression_model.Set(new(loss.MeanSquaredError), optimization.CreateAdaptiveMomentum(0.005, .001, 0.0000001, 0.9, 0.999, 0), new(accuracy.RegressionAccuracy))

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

	binary_categorical_model.Set(new(loss.BinaryCrossEntropy), optimization.CreateAdaptiveMomentum(1e-3, 5e-7, 1e-7, 0.9, 0.999, 0), new(accuracy.BinaryAccuracy))

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

	classification_model.Set(new(loss.CategoricalCrossEntropy), optimization.CreateAdaptiveMomentum(0.05, 5e-5, 1e-7, 0.9, 0.999, 0), new(accuracy.CategoricalAccuracy))

	classification_model.Finalize()

	classification_model.Train(datamodels.TrainingData{X, y}, datamodels.ValidationData{X_test, y_test}, 2, 1000, 1000)
}

func TestFashionMISTModel(t *testing.T) {
	training_data, testing_data := datasets.LoadFashionMNISTDataset(true)

	fashionMNIST_model := New()

	fashionMNIST_model.Add(layer.CreateLayer(training_data.X.RawMatrix().Cols, 512, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.ReLU))

	fashionMNIST_model.Add(layer.CreateLayer(512, 512, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.ReLU))

	fashionMNIST_model.Add(layer.CreateLayer(512, 10, 0, 0, 0, 0))
	fashionMNIST_model.Add(new(activation.SoftMax))

	fashionMNIST_model.Set(new(loss.CategoricalCrossEntropy), optimization.CreateAdaptiveMomentum(0.005, 5e-5, 1e-7, 0.9, 0.999, 0), new(accuracy.CategoricalAccuracy))

	fashionMNIST_model.Finalize()

	fashionMNIST_model.Train(training_data, testing_data, 1, 128, 100)

	//fashionMNIST_model.SaveParameters("fashionMNIST_model")
	modelDataProvider := new(ModelDataProvider)
	modelDataProvider.Save("fashionMNIST_model_full_3", fashionMNIST_model)
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
	fashionMNIST_model.LoadParameters("fashionMNIST_model_full_3")

	fashionMNIST_model.Evaluate(testing_data, 0)
}

func TestCANDatasetTraining(t *testing.T) {
	training_data, testing_data := datasets.LoadCANDataset(true)

	CAN_dataset_model := New()

	CAN_dataset_model.Add(layer.CreateLayer(training_data.X.RawMatrix().Cols, 896, 0, 1e-5, 0, 1e-5))
	CAN_dataset_model.Add(new(activation.ReLU))

	CAN_dataset_model.Add(layer.NewDropoutLayer(0.1))

	CAN_dataset_model.Add(layer.CreateLayer(896, 896, 0, 1e-5, 0, 1e-5))
	CAN_dataset_model.Add(new(activation.ReLU))

	CAN_dataset_model.Add(layer.CreateLayer(896, 2, 0, 1e-5, 0, 1e-5))
	CAN_dataset_model.Add(new(activation.SoftMax))

	CAN_dataset_model.Set(new(loss.CategoricalCrossEntropy), optimization.CreateAdaptiveMomentum(0.001, 1e-3, 1e-7, 0.9, 0.999, 1.0), new(accuracy.CategoricalAccuracy))

	CAN_dataset_model.Finalize()
	CAN_dataset_model.Train(training_data, testing_data, 5, 128, 10000)

	//	CAN_dataset_model.SaveParameters("CAN_dataset_model_parameters")

	modelDataProvider := new(ModelDataProvider)
	err := modelDataProvider.Save("CAN_dataset_model_full_shuffled_II", CAN_dataset_model)

	if err != nil {
		panic(err)
	}
}

func TestMisc(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	b := mat.NewDense(2, 2, []float64{6, 6, 6, 6})

	fmt.Println(mat.Formatted(a))
	a.Copy(b)
	fmt.Println(mat.Formatted(a))
}

func TestFashionMISTModelFromFile(t *testing.T) {
	_, testing_data := datasets.LoadFashionMNISTDataset(true)

	file, err := os.Open("./saved_models/fashionMNIST_model_full_3.json")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	modelDataProvider := new(ModelDataProvider)
	model, err := modelDataProvider.Load(file)
	if err != nil {
		t.Fatal(err)
	}

	model.Evaluate(testing_data, 0)
}

func TestModelInference(t *testing.T) {
	// get and save y_true
	label := 6
	can_data, y := datasets.LoadCANDatasetForInference("../../core/datasets/inference/single-2.csv", label)

	file, err := os.Open("./saved_models/CAN_dataset_model_full.json")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	modelDataProvider := new(ModelDataProvider)
	model, err := modelDataProvider.Load(file)
	if err != nil {
		t.Fatal(err)
	}

	confidences := model.Predict(can_data, 100)
	predictions := model.OutputLayerActivation.Predictions(confidences)

	_ = map[int]string{
		0: "attack-free",
		1: "combined-attacks",
		2: "DoS-attacks",
		3: "fuzzing-attacks",
		4: "gear-attacks",
		5: "interval-attacks",
		6: "rpm-attacks",
		7: "speed-attacks",
		8: "standstill-attacks",
		9: "systematic-attacks",
	}

	var zerCount, onesCount int
	for _, datum := range predictions.RawMatrix().Data {
		if datum == 0 {
			zerCount++
		} else {
			onesCount++
		}
	}

	fmt.Printf("attack free: %d, anomaly: %d", zerCount, onesCount)

	// Calculate Metrics
	var updPred, updYtrue []float64
	if label != 1 {
		for _, datum := range predictions.RawMatrix().Data {
			if datum == 0 {
				updPred = append(updPred, datum)
			} else {
				updPred = append(updPred, 1)
			}
		}

		for _, datum := range y {
			if datum == 0 {
				updYtrue = append(updYtrue, datum)
			} else {
				updYtrue = append(updYtrue, 1)
			}
		}
	}

	confusionMatrixMulti := metrics.ConfusionMatrix(updYtrue, updPred, 2)
	accuracyMulti, precisionMulti, recallMulti, f1ScoreMulti := metrics.CalculateMetrics(confusionMatrixMulti, 2)

	fmt.Println()
	fmt.Println("<== Metrics ==>")
	fmt.Println("Average Accuracy: ", stat.Mean(accuracyMulti, nil))
	fmt.Println("Average Precision: ", stat.Mean(precisionMulti, nil))
	fmt.Println("Average Recall: ", stat.Mean(recallMulti, nil))
	fmt.Println("Average F1: ", stat.Mean(f1ScoreMulti, nil))

	metrics.PlotConfusionMatrix(confusionMatrixMulti, 2, "rpm_confusion_matrix.png")
}

func TestModel_Predict(t *testing.T) {
	_, testing_data := datasets.LoadFashionMNISTDataset(false)

	file, err := os.Open("./saved_models/fashionMNIST_model_full.json")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	modelDataProvider := new(ModelDataProvider)
	model, err := modelDataProvider.Load(file)
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

	_, _ = datasets.LoadCANDataset(true)
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
