package main

import (
	"fmt"
	"github.com/saent-x/ids-nn/core"
	"gonum.org/v1/gonum/mat"
)

func main() {
	X, y := core.SpiralData(100, 3)

	fmt.Println(mat.Formatted(X))
	fmt.Println(y)

	fmt.Println("Plotting data...")
	core.PlotData(X, y, 100, 3)
	fmt.Println("Data plotted and saved to spiral.png")
}
