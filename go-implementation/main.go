package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"gonum.org/v1/gonum/mat"
)

func main() {
	X, _ := core.SpiralData(100, 3)

	layer_1 := core.CreateLayer(2, 3)
	layer_1.Forward(X)

	layer_2 := core.CreateLayer(3, 30)
	layer_2.Forward(layer_1.Output)

	activation_1 := new(core.ActivationReLU)
	activation_1.Forward(layer_1.Output)

	fmt.Println(mat.Formatted(activation_1.Output))

	//fmt.Println(mat.Formatted(X))
	//fmt.Println(y)
	//
	//fmt.Println("Plotting data...")
	//core.PlotData(X, y, 100, 3)
	//fmt.Println("Data plotted and saved to spiral.png")
}
