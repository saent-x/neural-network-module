package datasets

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/datamodels"
	"gonum.org/v1/gonum/mat"
	"os"
)

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
