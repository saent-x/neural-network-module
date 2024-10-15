package scaling

import "errors"

const (
	NEG_ONE_TO_POS_ONE = 0
	ZERO_TO_ONE        = 1
)

func Scale(scale_type int, value float64, maxValue float64) (float64, error) {
	if scale_type == NEG_ONE_TO_POS_ONE {
		return scale1(value, maxValue), nil
	} else if scale_type == ZERO_TO_ONE {
		return scale2(value, maxValue), nil
	} else {
		return 0, errors.New("scale type error")
	}
}

// scale1 func scales a value from -1 to 1
func scale1(value float64, maxValue float64) float64 {
	halfOfMax := maxValue / 2

	return (value - halfOfMax) / halfOfMax
}

// scale2 func scales a value from 0 to 1
func scale2(value float64, maxValue float64) float64 {
	return value / maxValue
}
