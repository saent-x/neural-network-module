package main

import (
	"fmt"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	softmax_output := mat.NewDense(1, 3, []float64{0.7, 0.1, 0.2}) // 0.1, 0.5, 0.4, 0.02, 0.9, 0.08
	class_targets := mat.NewDense(1, 3, []float64{0, 1, 1})

	//target_output := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 1, 0})

	// r, _ := softmax_output.Dims()
	var result []float64
	lo.ForEach(class_targets.RawMatrix().Data, func(item float64, index int) {

		fmt.Println(index)
		result = append(result, softmax_output.At(index, int(item)))
	})

}
