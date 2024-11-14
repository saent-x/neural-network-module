package main

import (
	"encoding/hex"
	"fmt"
	"github.com/saent-x/ids-nn/core/metrics"
	"log"
	"math"
	"strconv"

	"github.com/saent-x/ids-nn/core/accuracy"
	"github.com/saent-x/ids-nn/core/activation"
	"github.com/saent-x/ids-nn/core/datasets"
	"github.com/saent-x/ids-nn/core/layer"
	"github.com/saent-x/ids-nn/core/loss"
	"github.com/saent-x/ids-nn/core/model"
	"github.com/saent-x/ids-nn/core/optimization"
)

func main() {
	//RunMetrics()
	_, _ = datasets.LoadCANDataset(true)

}

func RunMetrics() {
	yTrueBinary := []float64{1, 0, 1, 1, 0, 1, 0, 0, 1, 0}
	yPredBinary := []float64{1, 0, 0, 1, 0, 1, 0, 1, 1, 0}
	numClassesBinary := 2

	confusionMatrixBinary := metrics.ConfusionMatrix(yTrueBinary, yPredBinary, numClassesBinary)
	fmt.Printf("Binary Confusion Matrix: %v\n", confusionMatrixBinary)

	accuracyBinary, precisionBinary, recallBinary, f1ScoreBinary := metrics.CalculateMetrics(confusionMatrixBinary, numClassesBinary)
	fmt.Printf("Binary Accuracy: %v\n", accuracyBinary)
	fmt.Printf("Binary Precision: %v\n", precisionBinary)
	fmt.Printf("Binary Recall: %v\n", recallBinary)
	fmt.Printf("Binary F1-Score: %v\n", f1ScoreBinary)

	metrics.PlotConfusionMatrix(confusionMatrixBinary, numClassesBinary, "binary_confusion_matrix_heatmap.png")

	// Example 2: Multi-class Classification
	yTrueMulti := []float64{1, 0, 2, 1, 2, 1, 0, 2, 1, 0}
	yPredMulti := []float64{1, 0, 2, 1, 1, 1, 0, 2, 2, 0}
	numClassesMulti := 3

	confusionMatrixMulti := metrics.ConfusionMatrix(yTrueMulti, yPredMulti, numClassesMulti)
	fmt.Printf("Multi-class Confusion Matrix: %v\n", confusionMatrixMulti)

	accuracyMulti, precisionMulti, recallMulti, f1ScoreMulti := metrics.CalculateMetrics(confusionMatrixMulti, numClassesMulti)
	fmt.Printf("Multi-class Accuracy: %v\n", accuracyMulti)
	fmt.Printf("Multi-class Precision: %v\n", precisionMulti)
	fmt.Printf("Multi-class Recall: %v\n", recallMulti)
	fmt.Printf("Multi-class F1-Score: %v\n", f1ScoreMulti)

	metrics.PlotConfusionMatrix(confusionMatrixMulti, numClassesMulti, "multi_confusion_matrix_heatmap.png")
}

func RunHex() {
	flValue, err := strconv.ParseInt("19D", 16, 64)
	if err != nil {
		log.Fatalf("Error converting hex to decimal: %v", err)
	}
	fmt.Println("19D -> ", float64(flValue))

	flValue2, err := strconv.ParseUint("5891235686", 16, 64)
	if err != nil {
		log.Fatalf("Error converting hex to decimal: %v", err)
	}

	fmt.Println("C0 00 3F FD 00 00 00 FF -> ", float64(flValue2))
}

func hexToDecimal(hexValues []string) []float64 {
	decimalValues := make([]float64, len(hexValues))
	for i, hex := range hexValues {
		decValue, err := strconv.ParseInt(hex, 16, 64)
		if err != nil {
			log.Fatalf("Error converting hex to decimal: %v", err)
		}
		decimalValues[i] = float64(decValue)
	}
	return decimalValues
}

func HexToFloat(hexStr string) (float32, float64) {
	// Pad the hex string to ensure it's 8 or 16 characters long
	for len(hexStr) < 8 {
		hexStr = "0" + hexStr
	}

	// Decode the hex string to bytes
	bytes, err := hex.DecodeString(hexStr)
	if err != nil {
		panic(err)
	}

	// Reverse the byte order (Go uses little-endian, but the input is big-endian)
	for i := 0; i < len(bytes)/2; i++ {
		bytes[i], bytes[len(bytes)-1-i] = bytes[len(bytes)-1-i], bytes[i]
	}

	// Convert to float32
	var float32Value float32
	if len(bytes) >= 4 {
		float32Value = math.Float32frombits(uint32(bytes[0]) | uint32(bytes[1])<<8 | uint32(bytes[2])<<16 | uint32(bytes[3])<<24)
	}

	// Convert to float64
	var float64Value float64
	if len(bytes) == 8 {
		float64Value = math.Float64frombits(uint64(bytes[0]) | uint64(bytes[1])<<8 | uint64(bytes[2])<<16 | uint64(bytes[3])<<24 |
			uint64(bytes[4])<<32 | uint64(bytes[5])<<40 | uint64(bytes[6])<<48 | uint64(bytes[7])<<56)
	}

	return float32Value, float64Value
}

func TestCANDatasetTraining() {
	training_data, testing_data := datasets.LoadCANDataset(true)

	CAN_dataset_model := model.New()

	CAN_dataset_model.Add(layer.CreateLayer(training_data.X.RawMatrix().Cols, 896, 0, 0, 0, 0))
	CAN_dataset_model.Add(new(activation.ReLU))

	CAN_dataset_model.Add(layer.NewDropoutLayer(0.1))

	CAN_dataset_model.Add(layer.CreateLayer(896, 896, 0, 0, 0, 0))
	CAN_dataset_model.Add(new(activation.ReLU))

	CAN_dataset_model.Add(layer.CreateLayer(896, 10, 0, 0, 0, 0))
	CAN_dataset_model.Add(new(activation.SoftMax))

	CAN_dataset_model.Set(new(loss.CategoricalCrossEntropy), optimization.CreateAdaptiveMomentum(0.005, 5e-5, 1e-7, 0.9, 0.999, 0), new(accuracy.CategoricalAccuracy))

	CAN_dataset_model.Finalize()
	CAN_dataset_model.Train(training_data, testing_data, 10, 2000, 10000)

	//	CAN_dataset_model.SaveParameters("CAN_dataset_model_parameters")

	modelDataProvider := new(model.ModelDataProvider)
	err := modelDataProvider.Save("CAN_dataset_model_full_shuffled", CAN_dataset_model)

	if err != nil {
		panic(err)
	}
}
