package datasets

import (
	"encoding/csv"
	"errors"
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/datamodels"
	"github.com/saent-x/ids-nn/core/scaling"
	"gonum.org/v1/gonum/mat"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
)

func LoadCANDataset(shuffle bool) (datamodels.TrainingData, datamodels.ValidationData) {
	x, y, err := ReadCAN_Folder("../../core/datasets/can-training-partial-sm")
	if err != nil {
		panic(err)
	}

	x, y, err = Oversample(x, y)
	if err != nil {
		panic(err)
	}

	// Convert data to mat.Dense
	X_mat := mat.NewDense(len(x), len(x[0]), nil)
	Y_mat := mat.NewDense(1, len(y), nil)

	if shuffle {
		shuffledIdxs := core.ShuffleSlice(core.GetRange(len(x)))
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			idx := shuffledIdxs[i]
			X_mat.SetRow(idx, x[i])
			Y_mat.Set(0, idx, y[i])
		}
	} else {
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			X_mat.SetRow(i, x[i])
			Y_mat.Set(0, i, y[i])
		}
	}

	training_data := datamodels.TrainingData{
		X: X_mat,
		Y: Y_mat,
	}

	// get validation file
	x_test, y_test, err := ReadCAN_Folder("../../core/datasets/can-testing-partial-sm")
	if err != nil {
		panic(err)
	}

	x_test, y_test, err = Oversample(x_test, y_test)
	if err != nil {
		panic(err)
	}

	// Convert data to mat.Dense
	X_mat_test := mat.NewDense(len(x_test), len(x_test[0]), nil)
	Y_mat_test := mat.NewDense(1, len(y_test), y_test)

	for i, row := range x_test {
		X_mat_test.SetRow(i, row)
	}

	testing_data := datamodels.ValidationData{
		X: X_mat_test,
		Y: Y_mat_test,
	}

	//fmt.Println(mat.Formatted(core.FirstN(training_data.X, 10)))

	if err = ScaleValues(training_data.X); err != nil {
		panic(err)
	}
	if err = ScaleValues(testing_data.X); err != nil {
		panic(err)
	}

	return training_data, testing_data
}

// Oversample oversamples the attack frames to match the number of normal frames while respecting time intervals
func Oversample(x [][]float64, y []float64) ([][]float64, []float64, error) {
	if len(x) != len(y) {
		return nil, nil, errors.New("error: x & y must have same nos of rows! ")
	}

	var normalFrames, attackFrames [][]float64
	var y_normalFrames, y_attackFrames []float64

	for idx, row := range x {
		if y[idx] == 1 {
			attackFrames = append(attackFrames, row)
			y_attackFrames = append(y_attackFrames, y[idx])
		} else {
			normalFrames = append(normalFrames, row)
			y_normalFrames = append(y_normalFrames, y[idx])
		}
	}

	// Calculate how many attack frames need to be added
	attackCount := len(attackFrames)
	normalCount := len(normalFrames)

	if attackCount == 0 {
		fmt.Println("No attack frames found. Dataset cannot be oversampled.")
		return nil, nil, nil
	}

	if attackCount >= normalCount {
		fmt.Println("The number of attack frames is already equal or greater than normal frames.")
		return nil, nil, nil
	}

	// Oversample attack frames while keeping time intervals realistic
	newAttackFrames := make([][]float64, 0, normalCount)
	newYFrames := make([]float64, 0, attackCount)

	for len(newAttackFrames) < normalCount {
		// For each duplicate, we will add a small delta to the time interval to simulate new attack frames
		for i := 0; i < attackCount && len(newAttackFrames) < normalCount; i++ {
			var duplicateFrame []float64

			originalFrame := attackFrames[i]
			duplicateFrame = append(duplicateFrame, originalFrame...)

			// Adjust the time interval of the duplicated frame slightly to preserve temporal structure
			timeDelta := math.Min(0.01, originalFrame[9]/10) // Add 1% to 10% of the original time interval
			duplicateFrame[9] += timeDelta

			newAttackFrames = append(newAttackFrames, duplicateFrame)
			newYFrames = append(newYFrames, 1)
		}
	}

	// Combine the normal and oversampled attack frames
	balancedDataset := append(normalFrames, newAttackFrames...)
	balancedYFrames := append(y_normalFrames, newYFrames...)

	return balancedDataset, balancedYFrames, nil
}

func ScaleValues(matrix *mat.Dense) error {
	_, cols := matrix.Dims()

	for i := 0; i < cols; i++ {
		col := matrix.ColView(i)
		maxValue := mat.Max(col)
		var scaledCol []float64

		for j := 0; j < col.Len(); j++ {
			scaledValue, err := scaling.Scale(scaling.NEG_ONE_TO_POS_ONE, col.AtVec(j), maxValue)
			if err != nil {
				return err
			}
			scaledCol = append(scaledCol, scaledValue)
		}
		matrix.SetCol(i, scaledCol)
	}

	return nil
}

func checkLabel(folder string) (float64, error) {
	// TODO: needs to be updated to reflect actual classes after tests
	//switch folder {
	//case "attack-free":
	//	return 0., nil
	//case "combined-attacks":
	//	return 1., nil
	//case "DoS-attacks":
	//	return 2., nil
	//case "fuzzing-attacks":
	//	return 3., nil
	//case "gear-attacks":
	//	return 4., nil
	//case "interval-attacks":
	//	return 5., nil
	//case "rpm-attacks":
	//	return 6., nil
	//case "speed-attacks":
	//	return 7., nil
	//case "standstill-attacks":
	//	return 8., nil
	//case "systematic-attacks":
	//	return 9., nil
	//default:
	//	return 0, errors.New("invalid folder!")
	//}

	switch folder {
	case "attack-free":
		return 0., nil
	default:
		return 1, errors.New("invalid folder!")
	}
}

func ReadCAN_Folder(folderPath string) ([][]float64, []float64, error) {
	var allData [][]float64
	var allAttackValues []float64

	entries, err := os.ReadDir(folderPath)
	if err != nil {
		return nil, nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			dataPath := filepath.Join(folderPath, entry.Name())
			//label, err := checkLabel(entry.Name()) // TBD: ignoring this for now, since we're using only two labels

			if err != nil {
				return nil, nil, err
			}

			x, y, err1 := ReadCSVFolder(dataPath, 0)
			if err1 != nil {
				return nil, nil, err1
			}

			allData = append(allData, x...)
			allAttackValues = append(allAttackValues, y...)
		}
	}

	return allData, allAttackValues, nil
}

func ReadCSVFolder(folderPath string, label float64) ([][]float64, []float64, error) {
	var allData [][]float64
	var allAttackValues []float64

	// Walk through the directory
	err := filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {

		if err != nil {
			return err
		}

		// Check if it's a CSV file
		if !info.IsDir() && filepath.Ext(path) == ".csv" {
			// Read the CSV file
			data, attackValues, err := ReadCSV(path, label)
			if err != nil {
				return fmt.Errorf("error reading file %s: %v", path, err)
			}

			// Append the data and attack values
			allData = append(allData, data...)
			allAttackValues = append(allAttackValues, attackValues...)
		}

		return nil
	})

	if err != nil {
		return nil, nil, fmt.Errorf("error walking through directory: %v", err)
	}

	return allData, allAttackValues, nil
}

func ReadCSV(filepath string, label float64) ([][]float64, []float64, error) {
	file, err := os.Open(filepath)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil, nil, err
	}
	defer file.Close()

	return readCSV(file)
}

func ReadCSVFile(file io.Reader) ([][]float64, []float64, error) {
	return readCSV(file)
}

func readCSV(file io.Reader) ([][]float64, []float64, error) {
	// Create a new CSV reader
	reader := csv.NewReader(file)

	// Read the header (and discard it)
	_, err := reader.Read()
	if err != nil {
		fmt.Println("Error reading header:", err)
		return nil, nil, err
	}

	var data [][]float64
	var attackValues []float64
	var prevTimestamp float64
	lines := 0

	// Read the file line by line
	for {
		record, err := reader.Read()
		if err != nil {
			break // End of file or error
		}

		row := make([]float64, 12)
		for i := 0; i < 4; i++ {
			if i == 0 {
				row[i], err = strconv.ParseFloat(record[i], 64)
				if err != nil {
					fmt.Printf("Error parsing float in row %d, column %d: %v\n", len(data)+1, i+1, err)
					return nil, nil, err
				}
			} else if i == 1 {
				val, err := strconv.ParseInt(record[i], 16, 64)
				if err != nil {
					log.Fatalf("Error converting hex to decimal: %v", err)
				}

				row[i] = float64(val)
			} else if i == 2 {
				if record[i] == "" {
					row[2] = 0
					row[3] = 0
					row[4] = 0
					row[5] = 0
					row[6] = 0
					row[7] = 0
					row[8] = 0
					row[9] = 0
				} else {
					hex := core.CleanHexString(record[i])

					vals, err := core.ParseDataField(hex)
					if err != nil {
						log.Fatalf("Error converting hex to decimal: %v", err)
					}

					row[2] = vals[0]
					row[3] = vals[1]
					row[4] = vals[2]
					row[5] = vals[3]
					row[6] = vals[4]
					row[7] = vals[5]
					row[8] = vals[6]
					row[9] = vals[7]
				}

			} else if i == 3 {
				row[11], err = strconv.ParseFloat(record[i], 64)
				if err != nil {
					fmt.Printf("Error parsing float in row %d, column %d: %v\n", len(data)+1, i+1, err)
					return nil, nil, err
				}
			}
		}

		if lines == 0 {
			row[10] = 0 // For the first row, no previous timestamp
		} else {
			row[10] = row[0] - prevTimestamp
		}

		prevTimestamp = row[0]

		data = append(data, row[1:11])
		attackValue := row[11]

		//if attackValue == 1 {
		//	attackValue = label
		//}

		attackValues = append(attackValues, attackValue) // Append last column to attackValues

		lines++
	}

	return data, attackValues, nil
}

func LoadFashionMNISTDataset(shuffle bool) (datamodels.TrainingData, datamodels.ValidationData) {
	train_dataset_path := "../../core/datasets/fashion_mnist_images/train"
	test_dataset_path := "../../core/datasets/fashion_mnist_images/test"

	train_data, err := os.ReadDir(train_dataset_path)
	if err != nil {
		panic(err)
	}

	test_data, err := os.ReadDir(test_dataset_path)
	if err != nil {
		panic(err)
	}

	X, y, err1 := core.SaveDataToSlice(train_data, train_dataset_path, shuffle)
	X_test, y_test, err2 := core.SaveDataToSlice(test_data, test_dataset_path, shuffle)

	if err1 != nil {
		panic(err)
	}
	if err2 != nil {
		panic(err)
	}

	return datamodels.TrainingData{X, y}, datamodels.ValidationData{X_test, y_test}
}

func LoadCANDatasetForInference(shuffle bool, filepath string) *mat.Dense {
	file, err := os.Open(filepath)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil
	}
	defer file.Close()

	x, _, err := ReadCSVFile(file)
	if err != nil {
		panic(err)
	}

	// Convert data to mat.Dense
	X_mat := mat.NewDense(len(x), len(x[0]), nil)

	if shuffle {
		shuffledIdxs := core.ShuffleSlice(core.GetRange(len(x)))
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			idx := shuffledIdxs[i]
			X_mat.SetRow(idx, x[i])
		}
	} else {
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			X_mat.SetRow(i, x[i])
		}
	}

	if err = ScaleValues(X_mat); err != nil {
		panic(err)
	}

	return X_mat
}

func LoadFashionMNISTDatasetForInference(shuffle bool) *mat.Dense {
	var X [][]float64
	data_path := "../../core/datasets/fashion_mnist_images/inference"

	inferenceData, err := os.ReadDir(data_path)
	if err != nil {
		panic(err)
	}

	for _, img := range inferenceData {
		if !img.IsDir() {
			imgBytes, err2 := core.ReadBytes(fmt.Sprintf("%s/%s", data_path, img.Name()), true, true)
			if err2 != nil {
				panic(err2)
			}

			X = append(X, imgBytes)
		}
	}

	X_mat := mat.NewDense(len(X), len(X[0]), nil)

	if shuffle {
		shuffledIdxs := core.ShuffleSlice(core.GetRange(len(X)))
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			idx := shuffledIdxs[i]
			X_mat.SetRow(idx, X[i])
		}
	} else {
		for i := 0; i < X_mat.RawMatrix().Rows; i++ {
			X_mat.SetRow(i, X[i])
		}
	}

	return X_mat
}
