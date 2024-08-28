package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/activation"
	"gonum.org/v1/gonum/mat"
)

func main() {
	_, _ = core.SpiralData(100, 3)

	softmax_output := mat.NewDense(1, 3, []float64{0.7, 0.1, 0.2})
	target_output := mat.NewDense(1, 3, []float64{1, 0, 0})

	loss_function_1 := new(core.LossFunction)

	loss_function_1.Calc(softmax_output, target_output)
	fmt.Println("loss: ", loss_function_1.Loss)
}

func main2(X *mat.Dense) {
	layer_1 := core.CreateLayer(2, 3)
	layer_2 := core.CreateLayer(3, 3)

	activation_1 := new(activation.ReLU)
	activation_2 := new(activation.SoftMax)

	layer_1.Forward(X)
	activation_1.Forward(layer_1.Output)
	layer_2.Forward(activation_1.Output)
	activation_2.Forward(layer_2.Output)

	r, c := activation_2.Output.Dims()

	last_5_rows := activation_2.Output.Slice(r-5, r, 0, c)

	fmt.Println(mat.Formatted(last_5_rows))
}
