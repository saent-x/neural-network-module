package core

import (
	"fmt"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"log"
	"math"
	"math/rand"
)

func linspace(start, stop float64, num int) []float64 {
	arr := make([]float64, num)
	step := (stop - start) / float64(num-1)
	for i := 0; i < num; i++ {
		arr[i] = start + step*float64(i)
	}
	return arr
}

func VerticalData(samples, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(samples*classes, 2, nil)
	y := make([]float64, samples*classes)

	for _, class_number := range lo.Range(classes) {
		ix := lo.RangeWithSteps(samples*class_number, samples*(class_number+1), 1)

		lo.ForEach(ix, func(item int, index int) {
			X.Set(item, 0, rand.NormFloat64()*0.1+float64(class_number)/3)
			X.Set(item, 1, rand.NormFloat64()*0.1+0.5)
			y[item] = float64(class_number)
		})
	}
	return X, mat.NewDense(1, len(y), y)
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

func PlotScatter(X *mat.Dense, samples, classes int, filepath string) bool {
	p := plot.New()

	p.Title.Text = fmt.Sprintf("%v Data", lo.PascalCase(filepath))
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

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
			pts[i].X = X.At(index, 0) // since Y has two features/ columns
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
	if err := p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s.png", filepath)); err != nil {
		log.Fatalf("could not save plot: %v", err)

		return false
	} else {
		return true
	}
}

func PlotLine(x []float64, y []float64) *plot.Plot {
	p := plot.New()

	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	XY_pts := make(plotter.XYs, len(x))

	for i := 0; i < len(XY_pts); i++ {
		XY_pts[i].X = x[i]
		XY_pts[i].Y = y[i]
	}

	s, err := plotter.NewLine(XY_pts)
	if err != nil {
		log.Fatalf("could not create line plot: %v", err)
	}
	s.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255} //red
	p.Add(s)

	return p
}

func SavePlot(p *plot.Plot, filepath string) bool {
	p.Title.Text = fmt.Sprintf("%v Data", lo.PascalCase(filepath))

	// Save the plot to a PNG file
	if err := p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s.png", filepath)); err != nil {
		log.Fatalf("could not save plot: %v", err)

		return false
	} else {
		return true
	}
}
