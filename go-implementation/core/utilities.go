package core

import (
	"image/color"
	"log"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func linspace(start, stop float64, num int) []float64 {
	arr := make([]float64, num)
	step := (stop - start) / float64(num-1)
	for i := 0; i < num; i++ {
		arr[i] = start + step*float64(i)
	}
	return arr
}

func SpiralData(samples, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(samples*classes, 2, nil)
	y := make([]float64, samples*classes)

	for classNumber := 0; classNumber < classes; classNumber++ {
		ixStart := samples * classNumber
		ixEnd := samples * (classNumber + 1)

		r := linspace(0.0, 1.0, samples)
		t := linspace(float64(classNumber*4), float64((classNumber+1)*4), samples)
		for i := range t {
			t[i] += rand.NormFloat64() * 0.2 // Adding Gaussian noise
		}

		for i := ixStart; i < ixEnd; i++ {
			rIndex := i - ixStart
			X.Set(i, 0, r[rIndex]*math.Sin(t[rIndex]*2.5))
			X.Set(i, 1, r[rIndex]*math.Cos(t[rIndex]*2.5))
			y[i] = float64(classNumber)
		}
	}

	return X, mat.NewDense(1, len(y), y)
}

func PlotData(X *mat.Dense, samples, classes int) bool {
	p := plot.New()

	p.Title.Text = "Spiral Data"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	// Define different colors for different classes
	colors := []color.RGBA{
		{R: 255, G: 0, B: 0, A: 255},   // Red
		{R: 0, G: 255, B: 0, A: 255},   // Green
		{R: 0, G: 0, B: 255, A: 255},   // Blue
		{R: 255, G: 255, B: 0, A: 255}, // Yellow
		{R: 255, G: 0, B: 255, A: 255}, // Magenta
	}

	for classNumber := 0; classNumber < classes; classNumber++ {
		pts := make(plotter.XYs, samples)
		for i := 0; i < samples; i++ {
			index := classNumber*samples + i
			pts[i].X = X.At(index, 0)
			pts[i].Y = X.At(index, 1)
		}

		s, err := plotter.NewScatter(pts)
		if err != nil {
			log.Fatalf("could not create scatter plot: %v", err)
		}
		s.GlyphStyle.Color = colors[classNumber%len(colors)]
		s.GlyphStyle.Radius = vg.Points(3)

		p.Add(s)
	}

	// Save the plot to a PNG file
	if err := p.Save(8*vg.Inch, 8*vg.Inch, "spiral.png"); err != nil {
		log.Fatalf("could not save plot: %v", err)

		return false
	} else {
		return true
	}

}
